"""AI2-THOR environment boundary for embodied-memory experiments.

The important security property in this module is that :class:`Observation` and
:class:`StepResult` never contain AI2-THOR metadata.  The high-level VLM should
only be called with ``result.rgb``.  Metadata remains available through an
explicitly named backend-only method for grounding and evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ActionFeedback:
    """The exact execution status returned by AI2-THOR."""

    lastActionSuccess: bool
    errorMessage: str


@dataclass(frozen=True)
class StepResult:
    """Safe result that may cross the environment/agent boundary."""

    rgb: np.ndarray
    feedback: ActionFeedback


class ThorEnvironment:
    """Thin wrapper around ``ai2thor.controller.Controller``.

    Args:
        scene: AI2-THOR scene name, for example ``FloorPlan1``.
        controller_kwargs: Additional arguments forwarded to ``Controller``.
        controller: Optional injected controller, useful for tests.

    Metadata is intentionally retained in ``_last_event``.  Never pass
    ``get_hidden_metadata()`` to a model prompt.
    """

    def __init__(
        self,
        scene: str = "FloorPlan1",
        *,
        controller_kwargs: Mapping[str, Any] | None = None,
        controller: Any | None = None,
    ) -> None:
        if controller is None:
            try:
                from ai2thor.controller import Controller
            except ImportError as exc:  # pragma: no cover - dependency specific
                raise RuntimeError(
                    "AI2-THOR is not installed. Run `pip install ai2thor`."
                ) from exc

            kwargs = {
                "scene": scene,
                "width": 640,
                "height": 480,
                "renderDepthImage": False,
                "renderInstanceSegmentation": False,
            }
            kwargs.update(dict(controller_kwargs or {}))
            self.controller = Controller(**kwargs)
            self._last_event = self.controller.last_event
        else:
            self.controller = controller
            self._last_event = controller.last_event

        self.scene = scene

    def reset(self, scene: str | None = None) -> StepResult:
        """Reset the simulator and return only RGB plus reset feedback."""

        self.scene = scene or self.scene
        self._last_event = self.controller.reset(self.scene)
        return self._safe_result(self._last_event)

    def observe(self) -> np.ndarray:
        """Return an RGB copy; this is the only observation intended for VLMs."""

        return np.asarray(self._last_event.frame).copy()

    def execute(self, command: Mapping[str, Any]) -> StepResult:
        """Execute one strict AI2-THOR command dictionary."""

        if not isinstance(command, Mapping) or not command.get("action"):
            raise ValueError("command must be a mapping with a non-empty 'action'")
        self._last_event = self.controller.step(**dict(command))
        return self._safe_result(self._last_event)

    def get_hidden_metadata(self) -> Mapping[str, Any]:
        """Return backend state for grounding/evaluation only.

        This method exists so that privileged state access is auditable.  Its
        result must not be included in the arguments to ``VLMAgent.step``.
        """

        return self._last_event.metadata

    def get_interactable_object_ids(self) -> list[str]:
        """Return visible object IDs suitable for the grounding module.

        AI2-THOR does not expose one universal ``interactable`` flag.  Visibility
        is therefore the conservative first-stage filter; the simulator remains
        the authority on reachability and physical preconditions.
        """

        objects = self.get_hidden_metadata().get("objects", [])
        return [
            str(obj["objectId"])
            for obj in objects
            if obj.get("visible") and obj.get("objectId")
        ]

    def close(self) -> None:
        """Release the Unity process."""

        self.controller.stop()

    @staticmethod
    def _safe_result(event: Any) -> StepResult:
        metadata = event.metadata
        feedback = ActionFeedback(
            lastActionSuccess=bool(metadata.get("lastActionSuccess", False)),
            errorMessage=str(metadata.get("errorMessage") or ""),
        )
        return StepResult(rgb=np.asarray(event.frame).copy(), feedback=feedback)
