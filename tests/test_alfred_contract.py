"""The ALFRED path exposes only the narrow RGB and action contract."""

import json

import numpy as np
import pytest

from alfred import runner
from alfred.runner import PROMPT, bbox_mask, rgb_change_fraction, validate_response


def test_interaction_bbox_and_mask_coordinates():
    decision = validate_response(
        json.dumps({"kind": "action", "action": "PickupObject", "bbox": [2, 3, 5, 7]})
    )
    mask = bbox_mask(decision["bbox"])
    assert mask.shape == (300, 300)
    assert np.count_nonzero(mask) == 12
    assert np.all(mask[3:7, 2:5] == 1)


@pytest.mark.parametrize(
    "response",
    [
        {"kind": "action", "action": "MoveBack"},
        {"kind": "action", "action": "CleanObject", "bbox": [0, 0, 5, 5]},
        {"kind": "action", "action": "PickupObject", "bbox": [0, 0, 301, 5]},
        {"kind": "action", "action": "PickupObject", "bbox": [5, 5, 5, 6]},
        {"kind": "action", "action": "PickupObject", "objectId": "Mug|0|0|0"},
        {"kind": "action", "action": "RotateLeft", "bbox": [0, 0, 5, 5]},
    ],
)
def test_forbidden_or_malformed_policy_output(response):
    with pytest.raises(ValueError):
        validate_response(json.dumps(response))


def test_prompt_has_no_state_placeholder():
    prompt = PROMPT.format(
        task="Place the mug",
        subtask="Pick it up",
        allowed="MoveAhead",
    )
    assert "metadata" not in prompt.lower()
    assert "objectId" not in prompt
    assert "goal_satisfied" not in prompt
    assert "{feedback}" not in prompt


def test_prompt_labels_current_and_historical_images_without_state_leakage():
    prompt = PROMPT.format(
        task="Place the mug", subtask="Find it", allowed="MoveAhead"
    )
    assert "Image 1 is the current" in prompt
    assert "older observations" in prompt
    assert "metadata" not in prompt.lower()
    assert "objectId" not in prompt


def test_rgb_change_fraction_is_visual_only():
    before = np.zeros((2, 2, 3), dtype=np.uint8)
    after = before.copy()
    after[0, 0, 1] = 9
    assert rgb_change_fraction(before, after) == pytest.approx(1 / 12)


def test_http_policy_request_contains_only_rgb_and_prompt(monkeypatch, tmp_path):
    observed = {}

    def fake_http(url, data):
        observed.update(data)
        return {"text": '{"kind":"complete"}', "tokens": 3}

    monkeypatch.setattr(runner, "http_json", fake_http)
    response, request_ref = runner.call_policy(
        "http://127.0.0.1:8001",
        [np.zeros((300, 300, 3), dtype=np.uint8)],
        "Pick up the mug",
        tmp_path,
        0,
    )
    assert set(observed) == {"images", "prompt"}
    assert observed["prompt"] == "Pick up the mug"
    assert len(observed["images"]) == 1
    assert response["tokens"] == 3
    assert (tmp_path / request_ref["path"]).is_file()
    assert (tmp_path / request_ref["response_path"]).is_file()
