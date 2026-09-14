"""Gradio dashboard wiring the complete Nexus Memory research loop.

Run with::

    python app.py

The default VLM is ``google/gemma-3-4b-it``. Set ``NEXUS_VLM_MODEL=demo`` only
when smoke-testing the UI without loading model weights.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import gradio as gr

from agent import DEFAULT_VLM_MODEL, DemoVLMBackend, Gemma3VLMBackend, VLMAgent
from environment import ActionFeedback, ThorEnvironment
from evaluator import TaskEvaluator
from memory import BaseMemory, HierarchicalMemory, MemFlatMemory
from translator import ActionTranslator, TranslationError

UIResult = tuple[Any, str, str, str, str, str, str]
SESSION_TTL_SECONDS = float(os.getenv("NEXUS_SESSION_TTL_SECONDS", "3600"))
LOGGER = logging.getLogger("nexus_memory.sessions")


@dataclass
class ResearchSession:
    environment: ThorEnvironment
    memory: BaseMemory
    agent: VLMAgent
    translator: ActionTranslator
    evaluator: TaskEvaluator
    global_goal: str
    stop_event: threading.Event = field(default_factory=threading.Event)
    operation_lock: threading.RLock = field(default_factory=threading.RLock)
    last_accessed: float = field(default_factory=time.monotonic)
    closed: bool = False

    def touch(self) -> None:
        self.last_accessed = time.monotonic()

    def close(self) -> None:
        """Stop and release the episode exactly once, after active work ends."""

        with self.operation_lock:
            if self.closed:
                return
            self.closed = True
            self.stop_event.set()
            try:
                self.environment.close()
            except Exception:  # pragma: no cover - simulator-specific shutdown
                LOGGER.exception("Failed to close an AI2-THOR session cleanly")


class SessionStore:
    """Process-local session registry; Gradio State contains only a safe ID."""

    def __init__(self) -> None:
        self._sessions: dict[str, ResearchSession] = {}
        self._lock = threading.RLock()

    def replace(self, old_id: str | None, session: ResearchSession) -> str:
        self.cleanup_expired()
        old: ResearchSession | None = None
        with self._lock:
            if old_id:
                old = self._sessions.pop(old_id, None)
            session_id = uuid.uuid4().hex
            self._sessions[session_id] = session
        if old:
            old.close()
        return session_id

    def get(self, session_id: str) -> ResearchSession:
        self.cleanup_expired()
        with self._lock:
            try:
                session = self._sessions[session_id]
            except KeyError as exc:
                raise gr.Error("Initialize an episode before stepping.") from exc
            session.touch()
            return session

    def remove(self, session_id: str) -> bool:
        """Remove and close one browser/API episode if it still exists."""

        if not session_id:
            return False
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        session.close()
        return True

    def cleanup_expired(self) -> int:
        """Close sessions idle beyond the configured TTL."""

        cutoff = time.monotonic() - SESSION_TTL_SECONDS
        with self._lock:
            expired_ids = [
                session_id
                for session_id, session in self._sessions.items()
                if session.last_accessed < cutoff
            ]
        return sum(self.remove(session_id) for session_id in expired_ids)

    def close_all(self) -> None:
        """Release every simulator when the dashboard process exits."""

        with self._lock:
            session_ids = list(self._sessions)
        for session_id in session_ids:
            self.remove(session_id)


SESSIONS = SessionStore()
atexit.register(SESSIONS.close_all)
_BACKENDS: dict[str, Any] = {}
_BACKENDS_LOCK = threading.Lock()


def _memory_for(name: str) -> BaseMemory:
    if name == "MEM-Flat":
        return MemFlatMemory()
    if name == "Hierarchical":
        return HierarchicalMemory()
    raise ValueError(f"Unknown memory architecture: {name}")


def _agent() -> VLMAgent:
    model_id = os.getenv("NEXUS_VLM_MODEL", DEFAULT_VLM_MODEL).strip()
    backend_key = model_id or DEFAULT_VLM_MODEL
    with _BACKENDS_LOCK:
        backend = _BACKENDS.get(backend_key)
        if backend is None:
            backend = (
                DemoVLMBackend()
                if backend_key.lower() == "demo"
                else Gemma3VLMBackend(backend_key)
            )
            _BACKENDS[backend_key] = backend
    return VLMAgent(backend)


def _feedback_text(feedback: ActionFeedback) -> str:
    return json.dumps(
        {
            "lastActionSuccess": feedback.lastActionSuccess,
            "errorMessage": feedback.errorMessage,
        },
        indent=2,
    )


def initialize(
    session_id: str | None, global_goal: str, architecture: str, scene: str
) -> tuple[str, Any, str, str, str, str, str, str]:
    if not global_goal.strip():
        raise gr.Error("Enter a global goal before initialization.")

    # Load/resolve the shared model before starting Unity so authentication or
    # model-loading failures cannot leave an unregistered simulator behind.
    agent = _agent()
    environment = ThorEnvironment(scene=scene.strip() or "FloorPlan1")
    memory = _memory_for(architecture)
    memory.reset(global_goal)
    session = ResearchSession(
        environment=environment,
        memory=memory,
        agent=agent,
        translator=ActionTranslator(),
        evaluator=TaskEvaluator(),
        global_goal=global_goal,
    )
    new_id = SESSIONS.replace(session_id, session)
    frame = environment.observe()
    evaluator_text = "Not yet evaluated | completion=0.000"
    backend_name = type(agent.backend).__name__
    return (
        new_id,
        frame,
        memory.get_context(),
        "",
        "{}",
        '{"lastActionSuccess": true, "errorMessage": ""}',
        evaluator_text,
        f"{agent.cost_summary()} | backend={backend_name}",
    )


def run_one_step(session_id: str, global_goal: str) -> UIResult:
    """Execute Agent -> Translator -> Environment -> Memory -> Evaluator."""

    session = SESSIONS.get(session_id)
    with session.operation_lock:
        if session.closed:
            raise gr.Error("This episode is closed. Initialize a new episode.")
        outputs = _run_one_step(session, global_goal)
        succeeded = session.evaluator.last_report.is_success
    if succeeded:
        SESSIONS.remove(session_id)
    return outputs


def _run_one_step(session: ResearchSession, global_goal: str) -> UIResult:
    """Run one step while the caller holds the session operation lock."""

    session.global_goal = global_goal

    # Privacy boundary: only RGB, model-visible memory, and the user goal enter
    # this call. The metadata lookup happens strictly after policy inference.
    rgb = session.environment.observe()
    proposed_memory, subtask = session.agent.step(
        rgb, session.memory.get_context(), global_goal
    )

    action: dict[str, Any] | None = None
    try:
        # The translator receives IDs only, not object state dictionaries.
        object_ids = session.environment.get_interactable_object_ids()
        action = session.translator.translate(subtask, object_ids)
        result = session.environment.execute(action)
        feedback = result.feedback
        rgb = result.rgb
    except TranslationError as exc:
        # A grounding failure is part of the trajectory and must enter memory,
        # just like a simulator-side physical precondition failure.
        feedback = ActionFeedback(False, f"Action grounding failed: {exc}")

    memory_text = session.memory.update_memory(
        proposed_memory,
        subtask,
        action,
        feedback.lastActionSuccess,
        feedback.errorMessage,
    )

    # Privileged access is isolated here and never passed back into the agent.
    success, score = session.evaluator.evaluate_task(
        global_goal, session.environment.get_hidden_metadata()
    )
    report = session.evaluator.last_report
    unsupported = (
        f" | unsupported={list(report.unsupported_clauses)}"
        if report.unsupported_clauses
        else ""
    )
    evaluator_text = f"success={success} | completion={score:.3f}{unsupported}"
    return (
        rgb,
        memory_text,
        subtask,
        json.dumps(action or {}, indent=2),
        _feedback_text(feedback),
        evaluator_text,
        session.agent.cost_summary(),
    )


def auto_run(session_id: str, global_goal: str, max_steps: int) -> Iterator[UIResult]:
    session = SESSIONS.get(session_id)
    session.stop_event.clear()
    try:
        for _ in range(int(max_steps)):
            if session.stop_event.is_set():
                break
            outputs = run_one_step(session_id, global_goal)
            yield outputs
            if session.evaluator.last_report.is_success:
                break
    finally:
        # Auto-run reaching success, stop, error, or max steps ends the episode.
        SESSIONS.remove(session_id)


def stop(session_id: str) -> str:
    session = SESSIONS.get(session_id)
    session.stop_event.set()
    return json.dumps(
        {
            "status": "stop_requested",
            "message": "Auto-run will end after the current model/action step.",
        },
        indent=2,
    )


def _delete_session(session_id: str) -> None:
    """Gradio State callback invoked when a browser/API session expires."""

    SESSIONS.remove(session_id)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Nexus Memory — AI2-THOR Research Dashboard") as demo:
        gr.Markdown(
            "# Nexus Memory\n"
            "Compare long-horizon memory architectures under a strict RGB-only "
            "agent observation policy. Evaluator state remains hidden."
        )
        session_state = gr.State(
            "",
            time_to_live=SESSION_TTL_SECONDS,
            delete_callback=_delete_session,
        )

        with gr.Row():
            global_goal = gr.Textbox(
                label="Global Goal", value="Slice the apple", scale=3
            )
            architecture = gr.Dropdown(
                ["MEM-Flat", "Hierarchical"],
                value="MEM-Flat",
                label="Memory Architecture",
                scale=1,
            )
            scene = gr.Textbox(label="AI2-THOR Scene", value="FloorPlan1", scale=1)

        with gr.Row():
            image = gr.Image(
                label="Live AI2-THOR RGB (agent observation)", type="numpy"
            )
            memory_state = gr.Textbox(label="Current Memory State", lines=18)

        with gr.Row():
            subtask = gr.Textbox(label="VLM Subtask (Text)")
            action = gr.Code(label="Translated THOR Action (JSON)", language="json")
            feedback = gr.Code(
                label="Environment Feedback (Success/Error)", language="json"
            )

        with gr.Row():
            evaluation = gr.Textbox(label="Task Success Status (hidden evaluator)")
            token_cost = gr.Textbox(label="Token Cost Tracker")

        with gr.Row():
            initialize_button = gr.Button("Initialize", variant="primary")
            step_button = gr.Button("Step")
            max_steps = gr.Slider(
                1, 100, value=20, step=1, label="Auto-Run max N steps"
            )
            auto_button = gr.Button("Auto-Run")
            stop_button = gr.Button("Stop", variant="stop")

        outputs = [
            image,
            memory_state,
            subtask,
            action,
            feedback,
            evaluation,
            token_cost,
        ]
        initialize_button.click(
            initialize,
            inputs=[session_state, global_goal, architecture, scene],
            outputs=[session_state, *outputs],
        )
        step_button.click(
            run_one_step, inputs=[session_state, global_goal], outputs=outputs
        )
        auto_button.click(
            auto_run,
            inputs=[session_state, global_goal, max_steps],
            outputs=outputs,
        )
        stop_button.click(stop, inputs=[session_state], outputs=[feedback], queue=False)

    return demo


if __name__ == "__main__":
    build_app().queue(default_concurrency_limit=4).launch()
