"""A small, memory-free AI2-THOR low-level agent loop.

The loop receives one subtask, observes the current RGB frame plus a deliberately
bounded view of current simulator metadata, and emits either one validated
AI2-THOR action or a completion token.  It has no long-term memory, planner,
subtask queue, evidence ledger, or skill state machine.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np


class VisionBackend(Protocol):
    def generate_images(
        self, images: list[np.ndarray], prompt: str
    ) -> tuple[str, int | None]: ...


class PolicyError(ValueError):
    """The model response cannot safely be executed."""


MOVEMENT_ACTIONS = {
    "MoveAhead",
    "MoveBack",
    "RotateLeft",
    "RotateRight",
    "LookUp",
    "LookDown",
}
OBJECT_ACTIONS = {
    "PickupObject",
    "PutObject",
    "OpenObject",
    "CloseObject",
    "ToggleObjectOn",
    "ToggleObjectOff",
    "SliceObject",
    "BreakObject",
    "CleanObject",
    "DirtyObject",
    "FillObjectWithLiquid",
    "EmptyLiquidFromObject",
}
ALLOWED_ACTIONS = MOVEMENT_ACTIONS | OBJECT_ACTIONS


POLICY_PROMPT = """ROLE: MEMORY_FREE_LOW_LEVEL_AGENT
Complete exactly one subtask in AI2-THOR. Use only the attached current RGB and
CURRENT_VISIBLE_METADATA. The metadata contains only visible objects, inventory,
camera orientation, and previous action feedback; it has no hidden objects,
world coordinates, path, evaluator answer, or memory.

One or more RGB images are attached in newest-to-oldest order. Image 1 is the
current observation. Older images are only short-term visual context from this
same subtask; never treat an older image as the current state.

Return exactly one JSON object and no commentary. Choose one schema:
{"kind":"action","action":"MoveAhead"}
{"kind":"action","action":"PickupObject","objectRef":"visible_0"}
{"kind":"action","action":"FillObjectWithLiquid",
 "objectRef":"visible_0","fillLiquid":"water"}
{"kind":"complete","reason":"current evidence proving the subtask is done"}
{"kind":"blocked","reason":"why no safe progress action exists"}

Allowed action strings (case-sensitive; use these exact API names only):
MoveAhead, MoveBack, RotateLeft, RotateRight, LookUp, LookDown, PickupObject,
PutObject, OpenObject, CloseObject, ToggleObjectOn, ToggleObjectOff, SliceObject,
BreakObject, CleanObject, DirtyObject, FillObjectWithLiquid,
EmptyLiquidFromObject.

Use complete only when the current RGB/metadata proves the subtask is already
finished. Use only an objectRef present in visible_objects. Never emit Pass,
Teleport, forceAction, coordinates, or more than one action. A successful
primitive does not by itself prove that a broader subtask is complete.

Before choosing an action, inspect last_action and last_action_success. If the
subtask requests exactly one primitive (for example "turn right once") and that
matching primitive just succeeded, return complete immediately. Never repeat a
successful one-time primitive.

Object identity is strict. An Egg never satisfies an Apple subtask, and another
similar object never substitutes for the requested type. A target being absent
from the current view or metadata is not evidence of completion or blockage.
When a requested target is absent, normally change the view with RotateRight or
RotateLeft and inspect again. Return blocked only when current evidence shows
that every allowed progress action is unsafe or impossible.
"""


VERIFY_PROMPT = """ROLE: SUBTASK_VERIFIER
The environment-action budget is exhausted. Decide whether the one subtask is
complete from only the attached final RGB and CURRENT_VISIBLE_METADATA. Return
exactly one JSON object and no commentary:
{"success":false,"reason":"brief evidence"}
Do not infer success merely because an action was attempted or succeeded.
"""


@dataclass(frozen=True)
class ActionRecord:
    step: int
    command: dict[str, Any]
    success: bool
    error: str


@dataclass(frozen=True)
class ModelCall:
    purpose: str
    image_count: int
    tokens: int | None
    latency_seconds: float
    response: str


@dataclass
class LoopResult:
    status: str
    success: bool
    reason: str
    environment_steps: int
    model_calls: int
    trajectory: list[ActionRecord] = field(default_factory=list)
    calls: list[ModelCall] = field(default_factory=list)
    final_event: Any = field(default=None, repr=False)

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "success": self.success,
            "reason": self.reason,
            "environment_steps": self.environment_steps,
            "model_calls": self.model_calls,
            "trajectory": [asdict(item) for item in self.trajectory],
            "calls": [asdict(item) for item in self.calls],
        }


class ImageHistory:
    """Bounded per-subtask RGB time series, read newest first."""

    def __init__(self, max_steps: int):
        if type(max_steps) is not int or not 1 <= max_steps <= 16:
            raise ValueError("image_history_steps must be an integer from 1 to 16")
        self.max_steps = max_steps
        self._frames: deque[np.ndarray] = deque(maxlen=max_steps)

    def append(self, image: np.ndarray) -> None:
        array = np.asarray(image)
        if array.ndim != 3:
            raise ValueError("observation image must be an HxWxC array")
        self._frames.append(array.copy())

    def newest_first(self) -> list[np.ndarray]:
        return [frame.copy() for frame in reversed(self._frames)]

    def __len__(self) -> int:
        return len(self._frames)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if np.isfinite(number):
            return round(number, 3)
    return None


def _object_type_from_id(object_id: Any) -> str | None:
    if not isinstance(object_id, str) or not object_id:
        return None
    return object_id.split("|", 1)[0]


def _visible_object_records(
    event: Any,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    metadata = getattr(event, "metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    raw_objects = [
        obj
        for obj in metadata.get("objects", [])
        if isinstance(obj, Mapping)
        and obj.get("visible")
        and isinstance(obj.get("objectId"), str)
        and isinstance(obj.get("objectType"), str)
    ]
    raw_objects.sort(key=lambda obj: (obj["objectType"], obj["objectId"]))
    visible_objects: list[dict[str, Any]] = []
    object_ids: dict[str, str] = {}
    for index, obj in enumerate(raw_objects):
        object_ref = f"visible_{index}"
        object_ids[object_ref] = obj["objectId"]
        item: dict[str, Any] = {
            "objectRef": object_ref,
            "type": obj["objectType"],
        }
        distance = _finite_number(obj.get("distance"))
        if distance is not None:
            item["distance"] = distance
        for capability in (
            "pickupable",
            "receptacle",
            "openable",
            "toggleable",
            "sliceable",
            "breakable",
            "canFillWithLiquid",
        ):
            if type(obj.get(capability)) is bool:
                item[capability] = obj[capability]
        for state in (
            "isPickedUp",
            "isOpen",
            "isToggled",
            "isSliced",
            "isBroken",
            "isDirty",
            "isFilledWithLiquid",
        ):
            if type(obj.get(state)) is bool:
                item[state] = obj[state]
        if isinstance(obj.get("fillLiquid"), str):
            item["fillLiquid"] = obj["fillLiquid"]
        visible_objects.append(item)
    return visible_objects, object_ids


def visible_metadata(event: Any) -> dict[str, Any]:
    """Return the only metadata supplied to the policy.

    The allow-list is intentional: hidden objects, positions, bounding boxes,
    reachable points, and goal/evaluator state cannot cross this boundary.
    Ephemeral references replace THOR IDs because those IDs contain coordinates.
    """

    metadata = getattr(event, "metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    visible_objects, _ = _visible_object_records(event)

    inventory = []
    for obj in metadata.get("inventoryObjects", []):
        if not isinstance(obj, Mapping):
            continue
        object_type = obj.get("objectType") or _object_type_from_id(obj.get("objectId"))
        if isinstance(object_type, str):
            inventory.append(object_type)

    agent = metadata.get("agent", {})
    if not isinstance(agent, Mapping):
        agent = {}
    rotation = agent.get("rotation", {})
    if not isinstance(rotation, Mapping):
        rotation = {}
    camera = {
        "rotation_y": _finite_number(rotation.get("y")),
        "camera_horizon": _finite_number(agent.get("cameraHorizon")),
        "standing": agent.get("isStanding")
        if type(agent.get("isStanding")) is bool
        else None,
    }
    camera = {key: value for key, value in camera.items() if value is not None}
    return {
        "visible_objects": visible_objects,
        "inventory_types": sorted(inventory),
        "camera": camera,
        "last_action": metadata.get("lastAction")
        if isinstance(metadata.get("lastAction"), str)
        else "",
        "last_action_success": bool(metadata.get("lastActionSuccess", True)),
        "last_action_error": str(metadata.get("errorMessage") or "")[:500],
    }


def parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*\n?(.*?)\n?```", text, flags=re.DOTALL | re.IGNORECASE
    )
    if fenced:
        text = fenced.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PolicyError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise PolicyError("response must be one JSON object")
    return payload


def validate_policy_output(
    payload: Mapping[str, Any],
    object_ids: Mapping[str, str],
    *,
    sub_task: str = "",
    inventory_types: tuple[str, ...] = (),
) -> tuple[str, dict[str, Any] | str]:
    payload = dict(payload)
    kind = payload.get("kind")
    if kind in ALLOWED_ACTIONS:
        payload = {"kind": "action", "action": kind} | {
            key: value for key, value in payload.items() if key != "kind"
        }
        kind = "action"
    if kind in {"complete", "blocked"}:
        if set(payload) != {"kind", "reason"} or not isinstance(
            payload.get("reason"), str
        ):
            raise PolicyError(f"{kind} requires only a string reason")
        return str(kind), str(payload["reason"]).strip()[:1000]
    if kind != "action":
        raise PolicyError("kind must be action, complete, or blocked")
    action = payload.get("action")
    if action not in ALLOWED_ACTIONS:
        raise PolicyError(f"unsupported action: {action!r}")
    command: dict[str, Any] = {"action": action}
    expected = {"kind", "action"}
    if action in OBJECT_ACTIONS:
        expected.add("objectRef")
        object_ref = payload.get("objectRef")
        if object_ref not in object_ids:
            raise PolicyError("objectRef must match one currently visible object")
        command["objectId"] = object_ids[object_ref]
        object_type = _object_type_from_id(object_ids[object_ref]) or ""
        if (
            action == "PickupObject"
            and sub_task
            and object_type.lower() not in sub_task.lower()
        ):
            raise PolicyError(
                f"cannot pick up {object_type}: it is not the requested subtask target"
            )
        if action == "PickupObject" and inventory_types:
            raise PolicyError(
                "cannot pick up another object while inventory is not empty: "
                + ", ".join(inventory_types)
            )
    if action == "FillObjectWithLiquid":
        expected.add("fillLiquid")
        liquid = payload.get("fillLiquid")
        if liquid not in {"water", "coffee", "wine"}:
            raise PolicyError("fillLiquid must be water, coffee, or wine")
        command["fillLiquid"] = liquid
    if set(payload) != expected:
        raise PolicyError("unexpected or missing response fields")
    return "action", command


def _generate(
    backend: VisionBackend,
    images: list[np.ndarray],
    prompt: str,
    purpose: str,
    calls: list[ModelCall],
) -> dict[str, Any]:
    started = time.perf_counter()
    raw, tokens = backend.generate_images(images, prompt)
    calls.append(
        ModelCall(purpose, len(images), tokens, time.perf_counter() - started, raw)
    )
    return parse_json_object(raw)


def _default_backend() -> VisionBackend:
    from inference import APIBackend

    return APIBackend(
        os.getenv("NEXUS_VLM_ENDPOINT", "http://127.0.0.1:8001"),
        token=os.getenv("NEXUS_VLM_TOKEN", ""),
    )


def run_agent_loop(
    controller: Any,
    sub_task: str,
    max_steps: int = 10,
    *,
    backend: VisionBackend | None = None,
    format_retries: int = 1,
    image_history_steps: int = 4,
) -> LoopResult:
    """Execute one memory-free subtask with at most ``max_steps`` actions."""

    if not isinstance(sub_task, str) or not sub_task.strip():
        raise ValueError("sub_task must be a non-empty string")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if type(format_retries) is not int or format_retries < 0:
        raise ValueError("format_retries must be a non-negative integer")
    backend = backend or _default_backend()
    event = controller.last_event
    image_history = ImageHistory(image_history_steps)
    image_history.append(event.frame)
    trajectory: list[ActionRecord] = []
    calls: list[ModelCall] = []

    for step in range(max_steps):
        semantic = visible_metadata(event)
        _, object_ids = _visible_object_records(event)
        prompt = (
            POLICY_PROMPT
            + "\nSUBTASK:\n"
            + sub_task.strip()
            + "\nCURRENT_VISIBLE_METADATA:\n"
            + json.dumps(semantic, ensure_ascii=False)
        )
        decision: tuple[str, dict[str, Any] | str] | None = None
        error = ""
        for _attempt in range(format_retries + 1):
            attempt_prompt = prompt
            if error:
                attempt_prompt += (
                    "\nYOUR PREVIOUS RESPONSE WAS REJECTED:\n"
                    + error
                    + "\nReturn a fresh valid JSON object. Nothing was executed."
                )
            try:
                payload = _generate(
                    backend,
                    image_history.newest_first(),
                    attempt_prompt,
                    "policy",
                    calls,
                )
                decision = validate_policy_output(
                    payload,
                    object_ids,
                    sub_task=sub_task,
                    inventory_types=tuple(semantic["inventory_types"]),
                )
                break
            except (PolicyError, OSError, RuntimeError, TimeoutError) as exc:
                error = f"{type(exc).__name__}: {exc}"[:1000]
        if decision is None:
            return LoopResult(
                "policy_error",
                False,
                error,
                len(trajectory),
                len(calls),
                trajectory,
                calls,
                event,
            )
        kind, value = decision
        if kind == "complete":
            return LoopResult(
                "succeeded",
                True,
                str(value),
                len(trajectory),
                len(calls),
                trajectory,
                calls,
                event,
            )
        if kind == "blocked":
            return LoopResult(
                "blocked",
                False,
                str(value),
                len(trajectory),
                len(calls),
                trajectory,
                calls,
                event,
            )
        command = value
        assert isinstance(command, dict)
        try:
            event = controller.step(**command)
        except Exception as exc:
            return LoopResult(
                "execution_error",
                False,
                f"{type(exc).__name__}: action outcome unknown",
                len(trajectory),
                len(calls),
                trajectory,
                calls,
                event,
            )
        image_history.append(event.frame)
        metadata = getattr(event, "metadata", {})
        trajectory.append(
            ActionRecord(
                step=step,
                command=dict(command),
                success=bool(metadata.get("lastActionSuccess", False)),
                error=str(metadata.get("errorMessage") or "")[:1000],
            )
        )

    semantic = visible_metadata(event)
    verify_prompt = (
        VERIFY_PROMPT
        + "\nSUBTASK:\n"
        + sub_task.strip()
        + "\nCURRENT_VISIBLE_METADATA:\n"
        + json.dumps(semantic, ensure_ascii=False)
    )
    try:
        payload = _generate(
            backend,
            image_history.newest_first(),
            verify_prompt,
            "verification",
            calls,
        )
        if set(payload) != {"success", "reason"}:
            raise PolicyError("verification requires exactly success and reason")
        if type(payload["success"]) is not bool or not isinstance(
            payload["reason"], str
        ):
            raise PolicyError("verification fields have invalid types")
    except (PolicyError, OSError, RuntimeError, TimeoutError) as exc:
        return LoopResult(
            "verification_error",
            False,
            f"{type(exc).__name__}: {exc}"[:1000],
            len(trajectory),
            len(calls),
            trajectory,
            calls,
            event,
        )
    success = payload["success"]
    return LoopResult(
        "succeeded" if success else "max_steps",
        success,
        payload["reason"].strip()[:1000],
        len(trajectory),
        len(calls),
        trajectory,
        calls,
        event,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sub_task")
    parser.add_argument("--scene", default="FloorPlan1")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--image-history-steps", type=int, default=4)
    parser.add_argument("--output")
    args = parser.parse_args()
    from ai2thor.controller import Controller

    controller = Controller(
        scene=args.scene,
        width=640,
        height=480,
        platform=os.getenv("NEXUS_THOR_PLATFORM", "CloudRendering"),
    )
    try:
        result = run_agent_loop(
            controller,
            args.sub_task,
            args.max_steps,
            image_history_steps=args.image_history_steps,
        )
        rendered = json.dumps(result.summary(), ensure_ascii=False, indent=2)
        if args.output:
            from pathlib import Path

            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
