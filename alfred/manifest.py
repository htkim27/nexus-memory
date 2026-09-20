"""Build the fixed FloorPlan15 manifest from official ALFRED trajectory JSONs."""

import argparse
import hashlib
import json
from pathlib import Path

TASKS = [
    (
        "wine_bottle",
        "pick_and_place_simple-WineBottle-None-DiningTable-15",
        "trial_T20190906_184006_967003",
    ),
    (
        "clean_mug",
        "pick_clean_then_place_in_recep-Mug-None-CoffeeMachine-15",
        "trial_T20190909_111443_363349",
    ),
    (
        "heated_apple_slice",
        "pick_heat_then_place_in_recep-AppleSliced-None-DiningTable-15",
        "trial_T20190907_223758_523581",
    ),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_manifest(data_root, split_file, archive_hash):
    split = json.loads(Path(split_file).read_text())
    tasks = []
    for name, task_id, trial in TASKS:
        relative = "{}/{}/{}/traj_data.json".format("train", task_id, trial)
        path = Path(data_root) / relative
        matches = [
            item for item in split["train"] if item["task"] == task_id + "/" + trial
        ]
        if not matches:
            raise ValueError("Missing official train split: " + relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        traj = json.loads(path.read_text())
        if traj["scene"]["scene_num"] != 15:
            raise ValueError("Unexpected scene: " + relative)
        high = traj["plan"]["high_pddl"]
        low = traj["plan"]["low_actions"]
        annotation = traj["turk_annotations"]["anns"][0]
        tasks.append(
            {
                "name": name,
                "split": "train",
                "task_id": task_id,
                "trial": trial,
                "repeat_idx": 0,
                "trajectory_path": relative,
                "trajectory_sha256": sha256(path),
                "instruction": annotation["task_desc"],
                "high_instructions": annotation["high_descs"],
                "pddl_params": traj["pddl_params"],
                "high_pddl": [
                    {
                        "high_idx": item["high_idx"],
                        "action": item["discrete_action"]["action"],
                    }
                    for item in high
                ],
                "low_actions": [
                    {
                        "high_idx": item["high_idx"],
                        "action": item["discrete_action"]["action"],
                    }
                    for item in low
                ],
                "scene_num": 15,
                "initial_scene_sha256": hashlib.sha256(
                    json.dumps(traj["scene"], sort_keys=True).encode()
                ).hexdigest(),
            }
        )
    return {
        "prototype": "ALFRED original-rule single-scene subtask oracle induction",
        "alfred_commit": "f91f4c0c96c7a29f33d0557f86b0a21035379b3b",
        "ai2thor_version": "2.1.0",
        "data_archive_sha256": archive_hash,
        "split_file_sha256": sha256(split_file),
        "tasks": tasks,
        "budgets": {
            "policy_actions": 1000,
            "execution_failures": 10,
            "exploration_episodes_per_task": 3,
            "validation_episodes_per_task": 3,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.data_root, args.split_file, sha256(args.archive))
    Path(args.output).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
