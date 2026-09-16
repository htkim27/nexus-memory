"""Minimal Gradio console for the vanilla low-level agent."""

from __future__ import annotations

import json
import os
import queue
import threading
from collections.abc import Iterator
from typing import Any

import gradio as gr
from ai2thor.controller import Controller

from inference import APIBackend
from observer import ObserverSnapshot, ThirdPersonController
from vanilla_agent import LoopResult, run_agent_loop


def _backend() -> APIBackend:
    return APIBackend(
        os.getenv("NEXUS_VLM_ENDPOINT", "http://127.0.0.1:8001"),
        token=os.getenv("NEXUS_VLM_TOKEN", ""),
    )


def _snapshot_output(
    snapshot: ObserverSnapshot,
    status: str,
    trajectory: list[dict[str, Any]],
) -> tuple[Any, Any, str, str]:
    if snapshot.command is not None:
        trajectory.append(
            {
                "command": snapshot.command,
                "success": snapshot.success,
                "error": snapshot.error,
            }
        )
    return (
        snapshot.first_person,
        snapshot.third_person,
        status,
        json.dumps(trajectory, ensure_ascii=False, indent=2),
    )


def run_episode(
    sub_task: str,
    scene: str,
    max_steps: int,
    image_history_steps: int,
) -> Iterator[tuple[Any, Any, str, str]]:
    """Stream observer snapshots while one isolated episode runs."""

    if not sub_task.strip():
        raise gr.Error("Enter a subtask.")
    messages: queue.Queue[tuple[str, Any]] = queue.Queue()
    controller = Controller(
        scene=scene.strip() or "FloorPlan1",
        width=640,
        height=480,
        platform=os.getenv("NEXUS_THOR_PLATFORM", "CloudRendering"),
    )
    observer: ThirdPersonController | None = None
    trajectory: list[dict[str, Any]] = []
    try:
        observer = ThirdPersonController(
            controller,
            callback=lambda snapshot: messages.put(("snapshot", snapshot)),
        )
        yield _snapshot_output(observer.initial_snapshot(), "initialized", trajectory)

        def worker() -> None:
            try:
                result = run_agent_loop(
                    observer,
                    sub_task,
                    int(max_steps),
                    backend=_backend(),
                    image_history_steps=int(image_history_steps),
                )
                messages.put(("result", result))
            except Exception as exc:
                messages.put(("error", exc))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        latest = observer.initial_snapshot()
        while True:
            kind, payload = messages.get()
            if kind == "snapshot":
                latest = payload
                yield _snapshot_output(latest, "running", trajectory)
                continue
            if kind == "error":
                raise gr.Error(f"{type(payload).__name__}: {payload}")
            result: LoopResult = payload
            yield _snapshot_output(
                latest,
                json.dumps(
                    {
                        "status": result.status,
                        "success": result.success,
                        "reason": result.reason,
                        "environment_steps": result.environment_steps,
                        "model_calls": result.model_calls,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                trajectory,
            )
            break
        thread.join()
    finally:
        if observer is not None:
            observer.stop()
        else:
            controller.stop()


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Nexus Vanilla Agent") as demo:
        gr.Markdown(
            "# Nexus Vanilla Agent\n"
            "Memory-free subtask execution with first-person and observer views."
        )
        with gr.Row():
            sub_task = gr.Textbox(
                label="Subtask", value="Find and open a drawer", scale=3
            )
            scene = gr.Textbox(label="Scene", value="FloorPlan1", scale=1)
        with gr.Row():
            max_steps = gr.Slider(1, 30, value=10, step=1, label="Max actions")
            history_steps = gr.Slider(
                1, 16, value=4, step=1, label="RGB history frames"
            )
            run_button = gr.Button("Run", variant="primary")
        with gr.Row():
            first_person = gr.Image(label="Agent first-person", type="numpy")
            third_person = gr.Image(label="Third-person observer", type="numpy")
        with gr.Row():
            status = gr.Code(label="Status", language="json")
            trajectory = gr.Code(label="Action trajectory", language="json")
        run_button.click(
            run_episode,
            inputs=[sub_task, scene, max_steps, history_steps],
            outputs=[first_person, third_person, status, trajectory],
        )
    return demo


if __name__ == "__main__":
    build_app().queue(default_concurrency_limit=1).launch(
        server_name=os.getenv("NEXUS_UI_HOST", "127.0.0.1"),
        server_port=int(os.getenv("NEXUS_UI_PORT", "7860")),
    )
