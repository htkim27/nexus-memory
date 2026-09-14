"""Ground natural-language subtasks into strict AI2-THOR actions."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from difflib import SequenceMatcher
from typing import Any


class TranslationError(ValueError):
    """Raised when a subtask cannot be grounded without guessing dangerously."""


LLMMapper = Callable[[str, Sequence[str]], str | dict[str, Any]]


class ActionTranslator:
    """Translate a subtask using deterministic rules or an injected small LLM.

    The optional ``llm_mapper`` receives only the natural-language subtask and
    the currently visible object IDs.  It never receives full simulator state.
    Its output is validated against an allow-list before execution.
    """

    OBJECT_ACTIONS = {
        "open": "OpenObject",
        "close": "CloseObject",
        "pick up": "PickupObject",
        "pickup": "PickupObject",
        "grab": "PickupObject",
        "take": "PickupObject",
        "put": "PutObject",
        "place": "PutObject",
        "toggle on": "ToggleObjectOn",
        "turn on": "ToggleObjectOn",
        "toggle off": "ToggleObjectOff",
        "turn off": "ToggleObjectOff",
        "slice": "SliceObject",
        "break": "BreakObject",
        "clean": "CleanObject",
        "dirty": "DirtyObject",
        "fill": "FillObjectWithLiquid",
        "empty": "EmptyLiquidFromObject",
    }
    SIMPLE_ACTIONS = {
        "move ahead": {"action": "MoveAhead"},
        "move forward": {"action": "MoveAhead"},
        "move back": {"action": "MoveBack"},
        "turn left": {"action": "RotateLeft"},
        "rotate left": {"action": "RotateLeft"},
        "turn right": {"action": "RotateRight"},
        "rotate right": {"action": "RotateRight"},
        "look up": {"action": "LookUp"},
        "look down": {"action": "LookDown"},
        "stop": {"action": "Pass"},
        "done": {"action": "Pass"},
    }
    ALLOWED_ACTIONS = set(OBJECT_ACTIONS.values()) | {
        value["action"] for value in SIMPLE_ACTIONS.values()
    }

    def __init__(self, llm_mapper: LLMMapper | None = None) -> None:
        self.llm_mapper = llm_mapper

    def translate(
        self, subtask: str, interactable_object_ids: Sequence[str]
    ) -> dict[str, Any]:
        """Return one validated AI2-THOR API command."""

        text = " ".join(subtask.strip().lower().split())
        if not text:
            raise TranslationError("The VLM returned an empty subtask.")

        if self.llm_mapper is not None:
            raw = self.llm_mapper(subtask, list(interactable_object_ids))
            command = json.loads(raw) if isinstance(raw, str) else dict(raw)
            return self._validate(command, interactable_object_ids)

        for phrase, command in sorted(
            self.SIMPLE_ACTIONS.items(), key=lambda item: -len(item[0])
        ):
            if phrase in text:
                return dict(command)

        for phrase, thor_action in sorted(
            self.OBJECT_ACTIONS.items(), key=lambda item: -len(item[0])
        ):
            if re.search(rf"\b{re.escape(phrase)}\b", text):
                object_id = self._best_object_id(text, interactable_object_ids)
                command: dict[str, Any] = {
                    "action": thor_action,
                    "objectId": object_id,
                }
                if thor_action == "FillObjectWithLiquid":
                    command["fillLiquid"] = self._liquid_from_text(text)
                return self._validate(command, interactable_object_ids)

        raise TranslationError(f"Unsupported subtask: {subtask!r}")

    @classmethod
    def build_llm_prompt(cls, subtask: str, object_ids: Sequence[str]) -> str:
        """Prompt template for a lightweight JSON-only grounding model."""

        return (
            "Map exactly one instruction to one AI2-THOR action. Return JSON only.\n"
            f"Allowed actions: {sorted(cls.ALLOWED_ACTIONS)}\n"
            f"Visible objectIds: {list(object_ids)}\n"
            f"Instruction: {subtask}\n"
            "For object actions, objectId must exactly match the supplied list."
        )

    @staticmethod
    def _object_label(object_id: str) -> str:
        # IDs normally look like ``Fridge|x|y|z``; synthetic test IDs may use _.
        return re.split(r"[|_]", object_id, maxsplit=1)[0].lower()

    def _best_object_id(self, text: str, object_ids: Sequence[str]) -> str:
        if not object_ids:
            raise TranslationError("No visible objects are available for grounding.")

        ranked: list[tuple[float, str]] = []
        words = set(re.findall(r"[a-z0-9]+", text))
        for object_id in object_ids:
            label = self._object_label(str(object_id))
            exact = 1.0 if label in text or label in words else 0.0
            fuzzy = max(
                [SequenceMatcher(None, label, word).ratio() for word in words] or [0.0]
            )
            ranked.append((exact * 2.0 + fuzzy, str(object_id)))

        score, object_id = max(ranked, key=lambda item: item[0])
        if score < 0.62:
            labels = sorted({self._object_label(str(item)) for item in object_ids})
            raise TranslationError(
                f"Could not identify a target object. Visible types: {labels}"
            )
        return object_id

    def _validate(
        self, command: dict[str, Any], object_ids: Sequence[str]
    ) -> dict[str, Any]:
        action = command.get("action")
        if action not in self.ALLOWED_ACTIONS:
            raise TranslationError(f"Disallowed or unknown THOR action: {action!r}")
        if action in self.OBJECT_ACTIONS.values():
            object_id = command.get("objectId")
            if object_id not in object_ids:
                raise TranslationError(
                    "Grounded objectId is not in the current interactable list."
                )
            allowed_keys = {"action", "objectId"}
            if action == "FillObjectWithLiquid":
                allowed_keys.add("fillLiquid")
                if command.get("fillLiquid") not in {"water", "coffee", "wine"}:
                    raise TranslationError("fillLiquid must be water, coffee, or wine.")
        else:
            allowed_keys = {"action"}
        unknown_keys = set(command) - allowed_keys
        if unknown_keys:
            raise TranslationError(f"Unexpected action arguments: {unknown_keys}")
        return command

    @staticmethod
    def _liquid_from_text(text: str) -> str:
        for liquid in ("water", "coffee", "wine"):
            if liquid in text:
                return liquid
        return "water"
