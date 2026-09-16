"""Third-person camera isolation tests."""

from __future__ import annotations

import numpy as np

from observer import ThirdPersonController


class Event:
    def __init__(self, action="Initialize", *, third_party=False):
        self.frame = np.full((3, 4, 3), 7, dtype=np.uint8)
        self.metadata = {
            "agent": {
                "position": {"x": 1.0, "y": 0.9, "z": 2.0},
                "rotation": {"x": 0.0, "y": 90.0, "z": 0.0},
            },
            "lastAction": action,
            "lastActionSuccess": True,
            "errorMessage": "",
            "actionReturn": 0 if action == "AddThirdPartyCamera" else None,
            "objects": [],
            "inventoryObjects": [],
        }
        self.third_party_camera_frames = (
            [np.full((3, 4, 3), 9, dtype=np.uint8)] if third_party else []
        )


class Controller:
    def __init__(self):
        self.last_event = Event()
        self.commands = []

    def step(self, **command):
        self.commands.append(command)
        is_camera = command["action"] in {
            "AddThirdPartyCamera",
            "UpdateThirdPartyCamera",
        }
        self.last_event = Event(command["action"], third_party=is_camera)
        return self.last_event

    def stop(self):
        pass


def test_observer_camera_actions_do_not_replace_policy_action_event():
    base = Controller()
    snapshots = []
    controller = ThirdPersonController(base, callback=snapshots.append)
    event = controller.step(action="RotateRight")
    assert event.metadata["lastAction"] == "RotateRight"
    assert controller.last_event is event
    assert [item["action"] for item in base.commands] == [
        "AddThirdPartyCamera",
        "RotateRight",
        "UpdateThirdPartyCamera",
    ]
    assert int(snapshots[0].third_person[0, 0, 0]) == 9
    assert snapshots[0].command == {"action": "RotateRight"}
