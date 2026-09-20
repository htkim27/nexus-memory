"""Produce tracked summaries from local-only ALFRED episode artifacts."""

import argparse
import base64
import hashlib
import json
import struct
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit_policy_requests(run_dir):
    files = sorted(run_dir.glob("policy_request_*.json"))
    errors = []
    max_images = 0
    for path in files:
        payload = json.loads(path.read_text())
        if set(payload) != {"images", "prompt"}:
            errors.append(f"{path.name}: unexpected request keys")
            continue
        images = payload["images"]
        max_images = max(max_images, len(images))
        if not 1 <= len(images) <= 5:
            errors.append(f"{path.name}: {len(images)} images")
        if not isinstance(payload["prompt"], str):
            errors.append(f"{path.name}: invalid prompt")
        for image in images:
            data = base64.b64decode(image)
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                errors.append(f"{path.name}: non-PNG image")
                continue
            if struct.unpack(">II", data[16:24]) != (300, 300):
                errors.append(f"{path.name}: image dimensions")
    return {
        "requests": len(files),
        "max_images": max_images,
        "passed": not errors,
        "errors": errors,
    }


def summarize(root, names):
    results = []
    for name in names:
        run = root / name
        summary_path = run / "summary.json"
        summary = json.loads(summary_path.read_text())
        final_path = run / "final_state.json"
        final = json.loads(final_path.read_text()) if final_path.exists() else {}
        initial_path = run / "initial_state.json"
        initial = json.loads(initial_path.read_text()) if initial_path.exists() else {}
        health_path = run / "model_health.json"
        health = json.loads(health_path.read_text()) if health_path.exists() else {}
        result = {
            "run": name,
            "artifact": "../runs/alfred/" + name + "/summary.json",
            "artifact_scope": "local-only",
            "summary_sha256": sha256(summary_path),
            "runner_sha256": summary.get("runner_sha256", "unknown"),
            "manifest_sha256": summary.get("manifest_sha256", "unknown"),
            "task": summary["task"],
            "mode": summary["mode"],
            "status": summary["status"],
            "official_goal_satisfied": (
                summary["success"] if "goal_conditions_met" in summary else "unknown"
            ),
            "goal_conditions_met": summary.get("goal_conditions_met", "unknown"),
            "policy_actions": summary["policy_actions"],
            "execution_failures": summary["execution_failures"],
            "model_calls": summary["model_calls"],
            "policy_action_budget": summary["policy_action_budget"],
            "execution_failure_budget": summary["execution_failure_budget"],
            "simulator_api_calls": summary.get("simulator_api_calls", "unknown"),
            "initial_state_sha256": sha256(initial_path)
            if initial_path.exists()
            else "unknown",
            "initial_semantic_sha256": summary.get(
                "initial_semantic_sha256", "unknown"
            ),
            "initial_agent": initial.get("agent", "unknown"),
            "initial_object_count": len(initial.get("objects", [])),
            "initial_cleaned_count": len(initial.get("cleaned_objects", [])),
            "initial_heated_count": len(initial.get("heated_objects", [])),
            "cleaned_objects_final": len(final.get("cleaned_objects", [])),
            "heated_objects_final": len(final.get("heated_objects", [])),
            "reason": summary["reason"],
            "model": summary.get("model", "unknown"),
            "model_revision": health.get("model_revision", "unknown"),
            "quantization": health.get("quantization", "unknown"),
            "decoding": health.get("decoding", "unknown"),
            "plan_file": summary.get("plan_file", "unknown"),
            "plan_sha256": summary.get("plan_sha256", "unknown"),
            "policy_request_audit": audit_policy_requests(run),
        }
        results.append(result)
    tasks = {}
    for task_name in ("wine_bottle", "clean_mug", "heated_apple_slice"):
        task_runs = [item for item in results if item["task"] == task_name]
        expert = [
            item
            for item in task_runs
            if item["mode"] == "expert" and item["run"].startswith("expert-")
        ]
        vlm = [item for item in task_runs if item["mode"] == "vlm"]
        tasks[task_name] = {
            "expert_replay_succeeded": bool(expert)
            and all(item["official_goal_satisfied"] for item in expert),
            "exploration_episodes": len(vlm),
            "successful_vlm_episodes": sum(
                item["official_goal_satisfied"] is True for item in vlm
            ),
            "validation_episodes": 0,
            "status": "unresolved",
        }
    return {
        "prototype_status": "incomplete",
        "oracle_episodes": [],
        "tasks": tasks,
        "evidence_type": (
            "single diagnostic rollouts and harness controls; no VLM oracle validation"
        ),
        "runs": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs/alfred")
    parser.add_argument("--output", default="alfred/results.json")
    parser.add_argument("names", nargs="*")
    args = parser.parse_args()
    root = Path(args.runs_root)
    names = args.names or sorted(
        path.name for path in root.iterdir() if (path / "summary.json").is_file()
    )
    data = summarize(root, names)
    Path(args.output).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
