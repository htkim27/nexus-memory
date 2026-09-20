"""ALFRED 2.1.0 replay and RGB-only VLM episodes.

Run with the isolated Python 3.7 environment.
"""

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

MOVE = {"MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown"}
INTERACT = {
    "PickupObject",
    "PutObject",
    "OpenObject",
    "CloseObject",
    "ToggleObjectOn",
    "ToggleObjectOff",
    "SliceObject",
}
ALLOWED = MOVE | INTERACT
PROMPT = """You are executing one ALFRED subtask using RGB images only.
Image 1 is the current 300x300 RGB observation.  Any following images are
older observations from this same subtask, newest to oldest.  They are visual
context only: use image 1 for an interaction box.
Task: {task}
Subtask: {subtask}
Return exactly one JSON object. Examples:
{{"kind":"action","action":"MoveAhead"}}
{{"kind":"action","action":"PickupObject","bbox":[x1,y1,x2,y2]}}
{{"kind":"complete"}} requests subtask verification.
Allowed actions: {allowed}.
Boxes use integer pixel coordinates in the current 300x300 RGB image;
x2/y2 are exclusive. Do not invent object IDs or simulator state, and do not
use Markdown or an action plan.  The wording of a subtask is a goal, not a
reason to repeat one verb.  Ground every next action in image 1.  A scan uses
at most four quarter-turns total; if the older images show a completed scan,
choose a visibly useful move, gaze change, interaction, or complete request
instead of another rotation.  Never retry a forward move when the current
image still shows the same obstruction.  Do not emit an interaction until its
target is visible in image 1 and its box tightly covers that visible target."""


def validate_response(raw):
    obj = json.loads(raw)
    if not isinstance(obj, dict) or obj.get("kind") not in (
        "action",
        "complete",
        "blocked",
    ):
        raise ValueError("invalid kind")
    if obj["kind"] != "action":
        return {"kind": obj["kind"]}
    action = obj.get("action")
    if action not in ALLOWED:
        raise ValueError("forbidden action")
    if action in MOVE:
        if set(obj) != {"kind", "action"}:
            raise ValueError("movement schema")
        return obj
    box = obj.get("bbox")
    if (
        set(obj) != {"kind", "action", "bbox"}
        or not isinstance(box, list)
        or len(box) != 4
        or any(type(x) is not int for x in box)
    ):
        raise ValueError("interaction schema")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 300 and 0 <= y1 < y2 <= 300):
        raise ValueError("bbox outside 300x300 image or empty")
    return obj


def bbox_mask(box):
    x1, y1, x2, y2 = box
    mask = np.zeros((300, 300), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def rgb_change_fraction(before, after):
    """Fraction of RGB scalar values that changed between two observations."""
    if before.shape != after.shape:
        raise ValueError("cannot compare RGB observations with different shapes")
    return float(np.count_nonzero(before != after)) / before.size


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_image(frame, path):
    Image.fromarray(np.uint8(frame)).save(str(path))
    return {"path": path.name, "sha256": digest(path)}


def state(env):
    metadata = env.last_event.metadata
    return {
        "agent": metadata.get("agent"),
        "inventory": metadata.get("inventoryObjects"),
        "objects": metadata.get("objects"),
        "cleaned_objects": sorted(env.cleaned_objects),
        "heated_objects": sorted(env.heated_objects),
        "cooled_objects": sorted(env.cooled_objects),
        "goal_satisfied": bool(env.get_goal_satisfied()),
        "goal_conditions_met": env.get_goal_conditions_met(),
        "subgoal_idx": env.get_subgoal_idx(),
    }


def semantic_initial_hash(snapshot):
    agent = snapshot["agent"] or {}
    selected = {
        "agent_position": agent.get("position"),
        "agent_rotation": agent.get("rotation"),
        "camera_horizon": agent.get("cameraHorizon"),
        "objects": sorted(
            [
                {
                    "id": obj.get("objectId"),
                    "open": obj.get("isOpen"),
                    "toggled": obj.get("isToggled"),
                    "dirty": obj.get("isDirty"),
                    "parent_receptacles": sorted(obj.get("parentReceptacles") or []),
                    "receptacle_contents": sorted(obj.get("receptacleObjectIds") or []),
                }
                for obj in snapshot["objects"]
            ],
            key=lambda item: item["id"],
        ),
        "cleaned_objects": snapshot["cleaned_objects"],
        "heated_objects": snapshot["heated_objects"],
        "cooled_objects": snapshot["cooled_objects"],
    }
    return hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()


def append_event(path, value):
    with open(path, "a") as out:
        out.write(json.dumps(value, default=str, ensure_ascii=False) + "\n")


def load_env(alfred_root, display):
    root = str(Path(alfred_root).resolve())
    sys.path[:0] = [root, str(Path(root) / "gen")]
    import gen.constants as constants

    constants.X_DISPLAY = display
    from ai2thor.controller import Controller
    from env.thor_env import ThorEnv

    original_step = getattr(Controller.step, "_oracle_original_step", Controller.step)

    def counted_step(self, action, *args, **kwargs):
        calls = getattr(self, "_oracle_api_calls", None)
        if calls is not None:
            calls.append(action)
        return original_step(self, action, *args, **kwargs)

    counted_step._oracle_original_step = original_step
    Controller.step = counted_step
    env = ThorEnv(x_display=display)
    env._oracle_api_calls = []
    return env


def setup(env, traj, alfred_root):
    scene = traj["scene"]
    env.reset("FloorPlan{}".format(scene["scene_num"]))
    env.restore_scene(
        scene["object_poses"], scene["object_toggles"], scene["dirty_and_empty"]
    )
    env.step(dict(scene["init_action"]))
    args = SimpleNamespace(
        reward_config=str(Path(alfred_root) / "models/config/rewards.json")
    )
    env.set_task(traj, args, reward_type="dense")


def http_json(url, data=None, timeout=120):
    request = Request(
        url,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def image_base64(frame):
    buffer = io.BytesIO()
    Image.fromarray(np.uint8(frame)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def call_policy(endpoint, images, prompt, output_dir, call_index):
    request = {"images": [image_base64(im) for im in images], "prompt": prompt}
    request_path = output_dir / f"policy_request_{call_index:04d}.json"
    request_path.write_text(json.dumps(request))
    started = time.time()
    response = http_json(endpoint.rstrip("/") + "/generate", request)
    if not isinstance(response, dict) or not isinstance(response.get("text"), str):
        raise ValueError("invalid /generate response")
    response_path = output_dir / f"policy_response_{call_index:04d}.json"
    response_path.write_text(json.dumps(response))
    return response, {
        "path": request_path.name,
        "sha256": digest(request_path),
        "response_path": response_path.name,
        "response_sha256": digest(response_path),
        "latency_seconds": time.time() - started,
    }


def run_episode(args):
    if not 1 <= args.max_policy_actions <= 1000:
        raise ValueError("max_policy_actions must be within official 1..1000 bound")
    if not 1 <= args.max_failures <= 10:
        raise ValueError("max_failures must be within official 1..10 bound")
    if (
        args.repeat_action_limit is not None
        and not 2 <= args.repeat_action_limit <= 1000
    ):
        raise ValueError("repeat_action_limit must be 2..1000")
    manifest = json.loads(Path(args.manifest).read_text())
    task = next(item for item in manifest["tasks"] if item["name"] == args.task)
    trajectory = Path(args.data_root) / task["trajectory_path"]
    if digest(trajectory) != task["trajectory_sha256"]:
        raise ValueError("trajectory hash mismatch")
    traj = json.loads(trajectory.read_text())
    plan = None
    if args.plan_file:
        plan_path = Path(args.plan_file)
        plan = json.loads(plan_path.read_text())
        if plan.get("task") != args.task:
            raise ValueError("plan task mismatch")
        if "overrides" in plan:
            base = traj["turk_annotations"]["anns"][0]["high_descs"]
            if not isinstance(plan["overrides"], dict):
                raise ValueError("invalid plan overrides")
            plan["subtasks"] = [
                {
                    "instruction": plan["overrides"].get(str(idx), text),
                    "source_high_idx": idx,
                }
                for idx, text in enumerate(base)
            ]
        if not isinstance(plan.get("subtasks"), list):
            raise ValueError("plan subtasks missing")
        if not plan["subtasks"] or any(
            not isinstance(item.get("instruction"), str)
            or type(item.get("source_high_idx")) is not int
            or not 0 <= item["source_high_idx"] < len(task["high_instructions"])
            for item in plan["subtasks"]
        ):
            raise ValueError("invalid subtask plan")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    events = output / "events.jsonl"
    summary = {
        "task": args.task,
        "mode": args.mode,
        "status": "interrupted",
        "success": False,
        "trajectory_sha256": task["trajectory_sha256"],
        "runner_sha256": digest(__file__),
        "manifest_sha256": digest(args.manifest),
        "python_version": sys.version.split()[0],
        "alfred_commit": manifest["alfred_commit"],
        "policy_actions": 0,
        "execution_failures": 0,
        "model_calls": 0,
        "simulator_api_calls": 0,
        "simulator_additional_calls": 0,
        "policy_action_budget": args.max_policy_actions,
        "execution_failure_budget": args.max_failures,
        "repeat_action_limit": args.repeat_action_limit,
        "image_history_steps": 4,
        "smooth_nav": True,
        "mask_px_sample": 1,
        "started_unix": time.time(),
        "display": args.display,
        "model": "unknown",
        "plan_file": args.plan_file or "official annotation 0 high_descs",
        "plan_sha256": digest(args.plan_file) if args.plan_file else "unknown",
        "reason": "not finished",
    }
    env = None
    try:
        if args.mode == "vlm":
            health = http_json(args.endpoint.rstrip("/") + "/health")
            (output / "model_health.json").write_text(json.dumps(health, indent=2))
            summary["model"] = health.get("model_id", "unknown")
            if not health.get("ready"):
                raise RuntimeError("VLM server not ready")
        env = load_env(args.alfred_root, args.display)
        setup(env, traj, args.alfred_root)
        import gen.constants as constants

        summary["render_settings"] = {
            "screen_width": constants.DETECTION_SCREEN_WIDTH,
            "screen_height": constants.DETECTION_SCREEN_HEIGHT,
            "render_image": constants.RENDER_IMAGE,
            "render_depth_image": constants.RENDER_DEPTH_IMAGE,
            "render_class_image": constants.RENDER_CLASS_IMAGE,
            "render_object_image": constants.RENDER_OBJECT_IMAGE,
            "visibility_distance": constants.VISIBILITY_DISTANCE,
        }
        before = state(env)
        (output / "initial_state.json").write_text(json.dumps(before, default=str))
        summary["initial_semantic_sha256"] = semantic_initial_hash(before)
        append_event(
            events,
            {
                "type": "initial",
                "goal": before["goal_satisfied"],
                "state_sha256": digest(output / "initial_state.json"),
            },
        )
        if args.mode == "early_complete":
            append_event(
                events,
                {
                    "type": "complete_request",
                    "subtask_idx": 0,
                    "official_goal_satisfied": before["goal_satisfied"],
                    "official_subgoal_idx": before["subgoal_idx"],
                },
            )
            summary.update(
                status="control_completed",
                reason="completion requested before any action",
            )
        elif args.mode == "expert":
            low = traj["plan"]["low_actions"]
            for index, item in enumerate(low):
                action = item["discrete_action"]
                name = action["action"]
                if index in args.skip_action_index:
                    append_event(
                        events,
                        {"type": "control_omission", "index": index, "action": name},
                    )
                    continue
                if (
                    args.omit_final_put
                    and index == len(low) - 1
                    and name == "PutObject"
                ):
                    append_event(
                        events,
                        {"type": "control_omission", "index": index, "action": name},
                    )
                    break
                compressed = action.get("args", {}).get("mask")
                mask = (
                    env.decompress_mask(compressed) if compressed is not None else None
                )
                if (
                    args.replace_final_put_with
                    and index == len(low) - 1
                    and name == "PutObject"
                ):
                    candidates = [
                        obj
                        for obj in env.last_event.metadata["objects"]
                        if obj.get("visible")
                        and obj.get("objectType") == args.replace_final_put_with
                    ]
                    if not candidates:
                        raise ValueError("wrong receptacle not visible")
                    target_id = candidates[0]["objectId"]
                    colors = [
                        color
                        for color, object_id in (
                            env.last_event.color_to_object_id.items()
                        )
                        if object_id == target_id
                    ]
                    if not colors:
                        raise ValueError("wrong receptacle has no segmentation")
                    pixels = np.array(env.last_event.instance_segmentation_frame)
                    mask = np.all(pixels == colors[0], axis=2).astype(np.uint8)
                    append_event(
                        events,
                        {
                            "type": "control_replacement",
                            "index": index,
                            "target_type": args.replace_final_put_with,
                            "target_id": target_id,
                        },
                    )
                prev = state(env)
                frame = np.uint8(env.last_event.frame).copy()
                rgb = save_image(frame, output / f"rgb_{index:04d}.png")
                call_start = len(env._oracle_api_calls)
                success, event, target, error, api = env.va_interact(
                    name, interact_mask=mask
                )
                api_calls = env._oracle_api_calls[call_start:]
                summary["simulator_api_calls"] += len(api_calls)
                summary["simulator_additional_calls"] += max(0, len(api_calls) - 1)
                summary["policy_actions"] += 1
                if not success:
                    summary["execution_failures"] += 1
                reward, reward_done = env.get_transition_reward()
                after = state(env)
                after_frame = np.uint8(env.last_event.frame).copy()
                before_ids = {obj["objectId"] for obj in prev["objects"]}
                after_ids = {obj["objectId"] for obj in after["objects"]}
                append_event(
                    events,
                    {
                        "type": "expert_action",
                        "index": index,
                        "high_idx": item["high_idx"],
                        "action": name,
                        "rgb": rgb,
                        "target_id": target,
                        "api_action": api,
                        "simulator_api_calls": api_calls,
                        "success": success,
                        "error": str(error),
                        "transition_reward": reward,
                        "transition_done": reward_done,
                        "rgb_change_fraction": rgb_change_fraction(frame, after_frame),
                        "created_object_ids": sorted(after_ids - before_ids),
                        "removed_object_ids": sorted(before_ids - after_ids),
                        "before": prev,
                        "after": after,
                    },
                )
                if not success:
                    summary.update(
                        status="failed",
                        reason=f"expert action {index}: {error}",
                        first_failure_index=index,
                    )
                    break
            else:
                summary.update(
                    status="expert_replay_completed",
                    reason="all expert actions executed",
                )
            if args.omit_final_put and summary["status"] == "interrupted":
                summary.update(
                    status="control_completed", reason="final PutObject omitted"
                )
        else:
            history = deque(maxlen=4)
            subtask_idx = 0
            repeated_action = None
            repeat_count = 0
            instructions = (
                plan["subtasks"]
                if plan is not None
                else [
                    {"instruction": text, "source_high_idx": idx}
                    for idx, text in enumerate(task["high_instructions"])
                ]
            )
            while (
                summary["policy_actions"] < args.max_policy_actions
                and summary["execution_failures"] < args.max_failures
            ):
                if subtask_idx >= len(instructions):
                    summary.update(
                        status="vlm_completed", reason="all subtasks requested complete"
                    )
                    break
                frame = np.uint8(env.last_event.frame).copy()
                if frame.shape[:2] != (300, 300):
                    raise ValueError("ALFRED RGB must be 300x300")
                images = [frame] + list(history)
                prompt = PROMPT.format(
                    task=task["instruction"],
                    subtask=instructions[subtask_idx]["instruction"],
                    allowed=", ".join(sorted(ALLOWED)),
                )
                response, request_ref = call_policy(
                    args.endpoint, images, prompt, output, summary["model_calls"]
                )
                summary["model_calls"] += 1
                try:
                    decision = validate_response(response["text"])
                except (ValueError, json.JSONDecodeError) as error:
                    repair = (
                        prompt + f"\nYour last response was invalid ({error}). "
                        "Return one valid JSON object."
                    )
                    response, request_ref = call_policy(
                        args.endpoint, images, repair, output, summary["model_calls"]
                    )
                    summary["model_calls"] += 1
                    decision = validate_response(response["text"])
                index = summary["policy_actions"]
                rgb = save_image(frame, output / f"rgb_{index:04d}.png")
                if decision["kind"] == "complete":
                    verified = state(env)
                    append_event(
                        events,
                        {
                            "type": "complete_request",
                            "subtask_idx": subtask_idx,
                            "policy_request": request_ref,
                            "rgb": rgb,
                            "official_goal_satisfied": verified["goal_satisfied"],
                            "official_subgoal_idx": verified["subgoal_idx"],
                        },
                    )
                    subtask_idx += 1
                    history.clear()
                    if verified["goal_satisfied"]:
                        summary.update(
                            status="vlm_completed", reason="official goal satisfied"
                        )
                        break
                    if (
                        verified["subgoal_idx"]
                        < instructions[subtask_idx - 1]["source_high_idx"]
                    ):
                        summary.update(
                            status="failed",
                            reason="premature subtask completion request",
                        )
                        break
                    continue
                if decision["kind"] == "blocked":
                    summary.update(status="failed", reason="policy blocked")
                    break
                name = decision["action"]
                repeat_count = repeat_count + 1 if name == repeated_action else 1
                repeated_action = name
                mask = bbox_mask(decision["bbox"]) if name in INTERACT else None
                mask_ref = None
                if mask is not None:
                    mask_path = output / f"mask_{index:04d}.png"
                    Image.fromarray(mask * 255).save(str(mask_path))
                    mask_ref = {"path": mask_path.name, "sha256": digest(mask_path)}
                prev = state(env)
                call_start = len(env._oracle_api_calls)
                success, event, target, error, api = env.va_interact(
                    name, interact_mask=mask
                )
                api_calls = env._oracle_api_calls[call_start:]
                summary["simulator_api_calls"] += len(api_calls)
                summary["simulator_additional_calls"] += max(0, len(api_calls) - 1)
                summary["policy_actions"] += 1
                if not success:
                    summary["execution_failures"] += 1
                reward, reward_done = env.get_transition_reward()
                after = state(env)
                after_frame = np.uint8(env.last_event.frame).copy()
                before_ids = {obj["objectId"] for obj in prev["objects"]}
                after_ids = {obj["objectId"] for obj in after["objects"]}
                append_event(
                    events,
                    {
                        "type": "policy_action",
                        "index": index,
                        "subtask_idx": subtask_idx,
                        "policy_request": request_ref,
                        "response": response,
                        "decision": decision,
                        "rgb": rgb,
                        "mask": mask_ref,
                        "target_id": target,
                        "api_action": api,
                        "simulator_api_calls": api_calls,
                        "success": success,
                        "error": str(error),
                        "transition_reward": reward,
                        "transition_done": reward_done,
                        "rgb_change_fraction": rgb_change_fraction(frame, after_frame),
                        "created_object_ids": sorted(after_ids - before_ids),
                        "removed_object_ids": sorted(before_ids - after_ids),
                        "before": prev,
                        "after": after,
                    },
                )
                history.appendleft(frame)
                if (
                    args.repeat_action_limit
                    and repeat_count >= args.repeat_action_limit
                ):
                    summary.update(
                        status="policy_loop",
                        reason=f"same action {name} repeated {repeat_count} times",
                    )
                    break
            if summary["status"] == "interrupted":
                summary.update(
                    status="budget_exhausted",
                    reason="policy action or execution failure budget",
                )
        final = state(env)
        (output / "final_state.json").write_text(json.dumps(final, default=str))
        summary["success"] = final["goal_satisfied"]
        summary["goal_conditions_met"] = final["goal_conditions_met"]
        summary["final_state_sha256"] = digest(output / "final_state.json")
        if (
            args.mode == "expert"
            and summary["status"] == "expert_replay_completed"
            and not summary["success"]
        ):
            summary.update(
                status="failed", reason="expert replay ended without official goal"
            )
        if args.mode == "vlm" and summary["success"]:
            summary["status"] = "vlm_success"
    except Exception as error:
        summary.update(status="failed", reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["ended_unix"] = time.time()
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n"
        )
        if env is not None:
            env.stop()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("expert", "vlm", "early_complete"))
    parser.add_argument(
        "--task",
        required=True,
        choices=("wine_bottle", "clean_mug", "heated_apple_slice"),
    )
    parser.add_argument("--manifest", default="alfred/tasks.json")
    parser.add_argument("--alfred-root", default="/home/htkim/dev/alfred-upstream")
    parser.add_argument("--data-root", default="runs/alfred/source/json_2.1.0")
    parser.add_argument(
        "--display", default=os.environ.get("DISPLAY", ":0").lstrip(":")
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8001")
    parser.add_argument("--output", required=True)
    parser.add_argument("--omit-final-put", action="store_true")
    parser.add_argument("--skip-action-index", type=int, action="append", default=[])
    parser.add_argument("--replace-final-put-with")
    parser.add_argument("--max-policy-actions", type=int, default=1000)
    parser.add_argument("--max-failures", type=int, default=10)
    parser.add_argument("--plan-file")
    parser.add_argument("--repeat-action-limit", type=int)
    args = parser.parse_args()
    print(json.dumps(run_episode(args), indent=2))


if __name__ == "__main__":
    main()
