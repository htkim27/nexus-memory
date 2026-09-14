"""Long-horizon text-memory interfaces and baseline implementations."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping
from typing import Any


def _compact(value: str, limit: int) -> str:
    value = " ".join((value or "").split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


_OBJECT_ID_PATTERN = re.compile(
    r"(?P<object_type>[A-Za-z][A-Za-z0-9]*)\|[^\s,;:'\"()\[\]{}]+"
)


def _object_type(object_id: Any) -> str:
    """Extract a human-readable type without retaining metadata coordinates."""

    return re.split(r"[|_]", str(object_id), maxsplit=1)[0]


def _safe_action_summary(action: Mapping[str, Any] | None) -> str:
    """Serialize an action for VLM memory without privileged object IDs."""

    if not action:
        return "NONE"
    safe: dict[str, Any] = {"action": action.get("action", "UNKNOWN")}
    if action.get("objectId"):
        safe["targetType"] = _object_type(action["objectId"])
    if action.get("fillLiquid"):
        safe["fillLiquid"] = action["fillLiquid"]
    return json.dumps(safe, sort_keys=True)


def _safe_feedback(error_message: str, action: Mapping[str, Any] | None) -> str:
    """Remove any AI2-THOR IDs from feedback before it reaches the VLM."""

    safe = error_message or "Unknown failure"
    if action and action.get("objectId"):
        object_id = str(action["objectId"])
        safe = safe.replace(object_id, _object_type(object_id))
    return _OBJECT_ID_PATTERN.sub(lambda match: match.group("object_type"), safe)


class BaseMemory(ABC):
    """Common interface used by the experimental runner."""

    @abstractmethod
    def reset(self, global_goal: str = "") -> None:
        """Clear episode-local state."""

    @abstractmethod
    def get_context(self) -> str:
        """Serialize model-visible memory."""

    @abstractmethod
    def update_memory(
        self,
        new_memory: str,
        subtask: str,
        action: Mapping[str, Any] | None,
        lastActionSuccess: bool,
        errorMessage: str = "",
    ) -> str:
        """Update memory with the proposal and actual environment feedback."""


class MemFlatMemory(BaseMemory):
    """MEM-inspired autoregressive flat-text memory.

    ``new_memory`` is treated as the VLM's compressed rewrite of prior memory.
    The just-executed transition is appended after that rewrite so physical
    failures cannot be omitted by the same model call that proposed the action.
    """

    def __init__(self, max_chars: int = 8_000) -> None:
        self.max_chars = max_chars
        self._text = ""

    def reset(self, global_goal: str = "") -> None:
        self._text = f"Goal: {_compact(global_goal, 1000)}" if global_goal else ""

    def get_context(self) -> str:
        return self._text

    def update_memory(
        self,
        new_memory: str,
        subtask: str,
        action: Mapping[str, Any] | None,
        lastActionSuccess: bool,
        errorMessage: str = "",
    ) -> str:
        base = _safe_feedback(new_memory.strip(), None) or self._text
        status = "SUCCESS" if lastActionSuccess else "FAILURE"
        record = (
            f"[{status}] subtask={_compact(subtask, 300)!r}; "
            f"action={_safe_action_summary(action)}"
        )
        if not lastActionSuccess:
            safe_error = _safe_feedback(errorMessage, action)
            record += f"; error={_compact(safe_error, 500)!r}"
            record += "; do not repeat unchanged—satisfy the precondition or replan"
        self._text = self._tail(f"{base}\n{record}".strip())
        return self._text

    def _tail(self, text: str) -> str:
        if len(text) <= self.max_chars:
            return text
        # Preserve a visible truncation marker and the newest evidence.
        marker = "[Earlier memory compressed/truncated]\n"
        return marker + text[-(self.max_chars - len(marker)) :]


class HierarchicalMemory(BaseMemory):
    """A functional placeholder for a NexusSum-style tiered memory.

    This establishes tier boundaries without claiming to implement a learned
    summarizer.  A later experiment can replace ``_roll_up`` with a controlled
    LLM summarization call while keeping the public interface unchanged.
    """

    def __init__(self, working_capacity: int = 6, episode_capacity: int = 20) -> None:
        self.working_capacity = working_capacity
        self.episode_capacity = episode_capacity
        self.global_summary = ""
        self.episode_summaries: deque[str] = deque(maxlen=episode_capacity)
        self.working: deque[str] = deque()

    def reset(self, global_goal: str = "") -> None:
        self.global_summary = (
            f"Goal: {_compact(global_goal, 1000)}" if global_goal else ""
        )
        self.episode_summaries.clear()
        self.working.clear()

    def get_context(self) -> str:
        sections = [f"GLOBAL\n{self.global_summary or '(empty)'}"]
        if self.episode_summaries:
            sections.append("ROLLED-UP EVENTS\n" + "\n".join(self.episode_summaries))
        sections.append("RECENT EVENTS\n" + ("\n".join(self.working) or "(empty)"))
        return "\n\n".join(sections)

    def update_memory(
        self,
        new_memory: str,
        subtask: str,
        action: Mapping[str, Any] | None,
        lastActionSuccess: bool,
        errorMessage: str = "",
    ) -> str:
        if new_memory.strip():
            self.global_summary = _compact(_safe_feedback(new_memory, None), 2500)
        status = "SUCCESS" if lastActionSuccess else "FAILURE"
        event = f"[{status}] {_compact(subtask, 240)} -> {_safe_action_summary(action)}"
        if not lastActionSuccess:
            safe_error = _safe_feedback(errorMessage, action)
            event += (
                f" | {_compact(safe_error, 400)}"
                " | retry only after changing preconditions"
            )
        self.working.append(event)
        if len(self.working) > self.working_capacity:
            self._roll_up()
        return self.get_context()

    def _roll_up(self) -> None:
        count = max(1, self.working_capacity // 2)
        batch = [self.working.popleft() for _ in range(count)]
        failures = [item for item in batch if item.startswith("[FAILURE]")]
        summary = f"Completed event batch ({len(batch)} events)."
        if failures:
            # Failures are retained verbatim because they encode preconditions.
            summary += " Important failures: " + " || ".join(failures)
        else:
            summary += " " + " | ".join(batch)
        self.episode_summaries.append(_compact(summary, 1500))
