"""Tests for the memory-free low-level loop."""

from __future__ import annotations

import json

import numpy as np

from vanilla_agent import (
    ImageHistory,
    run_agent_loop,
    validate_policy_output,
    visible_metadata,
)


class Event:
    def __init__(self, *, visible=True, success=True):
        self.frame = np.zeros((4, 5, 3), dtype=np.uint8)
        self.metadata = {
            "lastAction": "Initialize",
            "lastActionSuccess": success,
            "errorMessage": "" if success else "failed",
            "agent": {
                "position": {"x": 1, "y": 2, "z": 3},
                "rotation": {"x": 0, "y": 90, "z": 0},
                "cameraHorizon": 30,
                "isStanding": True,
            },
            "inventoryObjects": [],
            "objects": [
                {
                    "objectId": "Apple|secret-coordinates",
                    "objectType": "Apple",
                    "visible": visible,
                    "position": {"x": 9, "y": 8, "z": 7},
                    "distance": 1.25,
                    "pickupable": True,
                    "isPickedUp": False,
                },
                {
                    "objectId": "Knife|hidden",
                    "objectType": "Knife",
                    "visible": False,
                    "position": {"x": 4, "y": 5, "z": 6},
                },
            ],
            "reachablePositions": [{"x": 1, "z": 1}],
            "evaluatorSuccess": True,
        }


class Controller:
    def __init__(self):
        self.last_event = Event()
        self.commands = []

    def step(self, **command):
        self.commands.append(command)
        self.last_event = Event()
        self.last_event.frame.fill(len(self.commands))
        self.last_event.metadata["lastAction"] = command["action"]
        return self.last_event


class Backend:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []
        self.image_batches = []

    def generate_images(self, images, prompt):
        self.prompts.append(prompt)
        self.image_batches.append([image.copy() for image in images])
        return next(self.responses), 7


def test_image_history_is_bounded_copied_and_newest_first():
    history = ImageHistory(2)
    source = np.full((2, 2, 3), 1, dtype=np.uint8)
    history.append(source)
    source.fill(9)
    history.append(np.full((2, 2, 3), 2, dtype=np.uint8))
    history.append(np.full((2, 2, 3), 3, dtype=np.uint8))
    frames = history.newest_first()
    assert len(frames) == 2
    assert [int(frame[0, 0, 0]) for frame in frames] == [3, 2]


def test_visible_metadata_is_allow_listed():
    payload = visible_metadata(Event())
    encoded = json.dumps(payload)
    assert payload["visible_objects"][0]["type"] == "Apple"
    assert payload["visible_objects"][0]["objectRef"] == "visible_0"
    assert "Knife" not in encoded
    assert "secret-coordinates" not in encoded
    assert "position" not in encoded
    assert "reachablePositions" not in encoded
    assert "evaluatorSuccess" not in encoded


def test_privileged_or_hidden_action_is_rejected():
    object_ids = {"visible_0": "Apple|secret-coordinates"}
    for payload in (
        {"kind": "action", "action": "Teleport", "x": 0},
        {"kind": "action", "action": "PickupObject", "objectRef": "visible_9"},
        {
            "kind": "action",
            "action": "MoveAhead",
            "forceAction": True,
        },
    ):
        try:
            validate_policy_output(payload, object_ids)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe output was accepted")


def test_pickup_must_match_subtask_target_and_empty_hand():
    payload = {
        "kind": "action",
        "action": "PickupObject",
        "objectRef": "visible_0",
    }
    object_ids = {"visible_0": "Egg|secret"}
    for kwargs in (
        {"sub_task": "Pick up the apple"},
        {"sub_task": "Pick up the egg", "inventory_types": ("Knife",)},
    ):
        try:
            validate_policy_output(payload, object_ids, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid pickup was accepted")


def test_unambiguous_action_in_kind_is_normalized():
    kind, command = validate_policy_output({"kind": "RotateLeft"}, {})
    assert kind == "action"
    assert command == {"action": "RotateLeft"}


def test_complete_token_exits_without_action():
    controller = Controller()
    backend = Backend(['{"kind":"complete","reason":"apple is held"}'])
    result = run_agent_loop(controller, "Pick up the apple", backend=backend)
    assert result.success is True
    assert result.status == "succeeded"
    assert result.environment_steps == 0
    assert controller.commands == []


def test_action_then_complete():
    controller = Controller()
    backend = Backend(
        [
            '{"kind":"action","action":"PickupObject","objectRef":"visible_0"}',
            '{"kind":"complete","reason":"pickup succeeded"}',
        ]
    )
    result = run_agent_loop(
        controller, "Pick up the apple", max_steps=3, backend=backend
    )
    assert result.success is True
    assert result.environment_steps == 1
    assert controller.commands == [
        {"action": "PickupObject", "objectId": "Apple|secret-coordinates"}
    ]
    assert len(backend.image_batches[0]) == 1
    assert [int(image[0, 0, 0]) for image in backend.image_batches[1]] == [1, 0]


def test_max_steps_uses_final_observation_verifier():
    controller = Controller()
    backend = Backend(
        [
            '{"kind":"action","action":"MoveAhead"}',
            '{"success":false,"reason":"target is not reached"}',
        ]
    )
    result = run_agent_loop(controller, "Reach the apple", max_steps=1, backend=backend)
    assert result.status == "max_steps"
    assert result.success is False
    assert result.environment_steps == 1
    assert result.model_calls == 2


def test_invalid_format_retries_without_executing_it():
    controller = Controller()
    backend = Backend(
        [
            "not-json",
            '{"kind":"action","action":"RotateRight"}',
            '{"success":true,"reason":"desired view reached"}',
        ]
    )
    result = run_agent_loop(
        controller, "Turn right", max_steps=1, backend=backend, format_retries=1
    )
    assert result.success is True
    assert controller.commands == [{"action": "RotateRight"}]
    assert "PREVIOUS RESPONSE WAS REJECTED" in backend.prompts[1]
