"""UI-only third-person camera isolated from the agent observation stream."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ObserverSnapshot:
    first_person: np.ndarray
    third_person: np.ndarray
    command: dict[str, Any] | None
    success: bool
    error: str


SnapshotCallback = Callable[[ObserverSnapshot], None]


class ThirdPersonController:
    """Proxy a Controller while maintaining a UI-only chase camera.

    Camera actions are sent to the underlying controller after each physical
    agent action. The original physical-action event is returned to the policy,
    so ``UpdateThirdPartyCamera`` never becomes policy feedback.
    """

    def __init__(
        self,
        controller: Any,
        *,
        callback: SnapshotCallback | None = None,
        distance: float = 1.5,
        height: float = 1.2,
        pitch: float = 25.0,
        field_of_view: float = 90.0,
    ) -> None:
        self.controller = controller
        self.callback = callback
        self.distance = float(distance)
        self.height = float(height)
        self.pitch = float(pitch)
        self.field_of_view = float(field_of_view)
        self.last_event = controller.last_event
        camera_event = self._camera_event(self.last_event, add=True)
        action_return = camera_event.metadata.get("actionReturn")
        self.camera_id = action_return if isinstance(action_return, int) else 0
        self.third_person = self._third_person_frame(camera_event)

    def _camera_pose(
        self, agent_event: Any
    ) -> tuple[dict[str, float], dict[str, float]]:
        agent = agent_event.metadata.get("agent", {})
        position = agent.get("position", {})
        rotation = agent.get("rotation", {})
        yaw = float(rotation.get("y", 0.0))
        radians = math.radians(yaw)
        camera_position = {
            "x": float(position.get("x", 0.0)) - math.sin(radians) * self.distance,
            "y": float(position.get("y", 0.0)) + self.height,
            "z": float(position.get("z", 0.0)) - math.cos(radians) * self.distance,
        }
        camera_rotation = {"x": self.pitch, "y": yaw, "z": 0.0}
        return camera_position, camera_rotation

    def _camera_event(self, agent_event: Any, *, add: bool) -> Any:
        position, rotation = self._camera_pose(agent_event)
        command: dict[str, Any] = {
            "action": "AddThirdPartyCamera" if add else "UpdateThirdPartyCamera",
            "position": position,
            "rotation": rotation,
            "fieldOfView": self.field_of_view,
        }
        if not add:
            command["thirdPartyCameraId"] = self.camera_id
        return self.controller.step(**command)

    @staticmethod
    def _third_person_frame(event: Any) -> np.ndarray:
        frames = getattr(event, "third_party_camera_frames", None) or []
        if not frames:
            raise RuntimeError("AI2-THOR did not return a third-party camera frame")
        return np.asarray(frames[-1]).copy()

    def initial_snapshot(self) -> ObserverSnapshot:
        return ObserverSnapshot(
            np.asarray(self.last_event.frame).copy(),
            self.third_person.copy(),
            None,
            True,
            "",
        )

    def step(self, **command: Any) -> Any:
        agent_event = self.controller.step(**command)
        camera_event = self._camera_event(agent_event, add=False)
        self.third_person = self._third_person_frame(camera_event)
        self.last_event = agent_event
        snapshot = ObserverSnapshot(
            np.asarray(agent_event.frame).copy(),
            self.third_person.copy(),
            dict(command),
            bool(agent_event.metadata.get("lastActionSuccess", False)),
            str(agent_event.metadata.get("errorMessage") or ""),
        )
        if self.callback:
            self.callback(snapshot)
        return agent_event

    def stop(self) -> None:
        self.controller.stop()
