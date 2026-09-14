"""Privileged task evaluation over hidden AI2-THOR metadata."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GoalCondition:
    object_type: str
    state_key: str
    expected: Any = True
    quantifier: str = "all"
    receptacle_type: str | None = None


@dataclass(frozen=True)
class EvaluationReport:
    is_success: bool
    completion_score: float
    satisfied: int
    total: int
    unsupported_clauses: tuple[str, ...] = ()


class TaskEvaluator:
    """Evaluate goal predicates without exposing metadata to the VLM.

    For reproducible experiments, prefer passing an explicit list of
    :class:`GoalCondition` objects.  The natural-language parser supports a
    deliberately small template vocabulary and fails closed on unknown clauses.
    """

    STATE_PATTERNS = {
        "clean": ("isClean", True),
        "dirty": ("isDirty", True),
        "slice": ("isSliced", True),
        "sliced": ("isSliced", True),
        "open": ("isOpen", True),
        "closed": ("isOpen", False),
        "on": ("isToggled", True),
        "off": ("isToggled", False),
        "break": ("isBroken", True),
        "broken": ("isBroken", True),
        "fill": ("isFilledWithLiquid", True),
        "filled": ("isFilledWithLiquid", True),
        "empty": ("isFilledWithLiquid", False),
        "pick up": ("isPickedUp", True),
        "picked up": ("isPickedUp", True),
    }

    def __init__(
        self, task_registry: Mapping[str, Sequence[GoalCondition]] | None = None
    ):
        self.task_registry = dict(task_registry or {})
        self.last_report = EvaluationReport(False, 0.0, 0, 0)

    def evaluate_task(
        self, global_goal: str, current_metadata: Mapping[str, Any]
    ) -> tuple[bool, float]:
        """Return ``(is_success, completion_score)`` in the range [0, 1]."""

        conditions = list(self.task_registry.get(global_goal, ()))
        unsupported: list[str] = []
        if not conditions:
            conditions, unsupported = self._parse_goal(global_goal)

        results = [
            self._condition_met(condition, current_metadata.get("objects", []))
            for condition in conditions
        ]
        # Each unsupported clause is an unsatisfied condition. This prevents a
        # partial parse from yielding a false positive for a compound task.
        total = len(results) + len(unsupported)
        satisfied = sum(results)
        score = satisfied / total if total else 0.0
        success = total > 0 and satisfied == total
        self.last_report = EvaluationReport(
            success, score, satisfied, total, tuple(unsupported)
        )
        return success, score

    def _parse_goal(self, goal: str) -> tuple[list[GoalCondition], list[str]]:
        clauses = [
            part.strip(" .")
            for part in re.split(r"\s*(?:,|\band\b|\bthen\b)\s*", goal.lower())
            if part.strip(" .")
        ]
        conditions: list[GoalCondition] = []
        unsupported: list[str] = []
        for clause in clauses:
            condition = self._parse_clause(clause)
            if condition is None:
                unsupported.append(clause)
            else:
                conditions.append(condition)
        return conditions, unsupported

    def _parse_clause(self, clause: str) -> GoalCondition | None:
        placement = re.search(
            r"\b(?:put|place|store)\s+(?:the\s+|all\s+|every\s+)?"
            r"(?P<object>[a-z][a-z0-9]*)\s+(?:in|inside|on)\s+(?:the\s+|a\s+)?"
            r"(?P<receptacle>[a-z][a-z0-9]*)\b",
            clause,
        )
        if placement:
            quantifier = "all" if re.search(r"\b(?:all|every)\b", clause) else "any"
            return GoalCondition(
                self._singularize(placement.group("object")),
                "__in_receptacle__",
                True,
                quantifier,
                self._singularize(placement.group("receptacle")),
            )

        state_match: tuple[str, Any] | None = None
        state_word = ""
        for candidate, state in sorted(
            self.STATE_PATTERNS.items(), key=lambda item: -len(item[0])
        ):
            if re.search(rf"\b{re.escape(candidate)}\b", clause):
                state_match, state_word = state, candidate
                break
        if state_match is None:
            return None

        # Handles templates such as "clean all plates", "the apple is sliced",
        # and "open the fridge". ObjectType matching is case-insensitive.
        ignored_words = (
            rf"is|are|be|make|make sure|the|a|an|all|every|objects?|"
            rf"{re.escape(state_word)}"
        )
        scrubbed = re.sub(rf"\b(?:{ignored_words})\b", " ", clause)
        tokens = re.findall(r"[a-z][a-z0-9]*", scrubbed)
        if not tokens:
            return None
        object_type = self._singularize(tokens[-1])
        quantifier = "all" if re.search(r"\b(?:all|every)\b", clause) else "any"
        return GoalCondition(object_type, state_match[0], state_match[1], quantifier)

    @staticmethod
    def _singularize(word: str) -> str:
        irregular = {"knives": "knife", "dishes": "dish", "glasses": "glass"}
        if word in irregular:
            return irregular[word]
        if word.endswith("ies"):
            return word[:-3] + "y"
        if word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
        return word

    @staticmethod
    def _condition_met(
        condition: GoalCondition, objects: Sequence[Mapping[str, Any]]
    ) -> bool:
        candidates = [
            obj
            for obj in objects
            if str(obj.get("objectType", "")).lower() == condition.object_type.lower()
        ]
        if not candidates:
            return False
        if condition.state_key == "__in_receptacle__":
            receptacle = str(condition.receptacle_type).lower()
            checks = [
                any(
                    str(parent).lower().startswith(receptacle + "|")
                    for parent in (obj.get("parentReceptacles") or [])
                )
                for obj in candidates
            ]
        elif condition.state_key == "isClean":
            # AI2-THOR commonly represents cleanliness as the inverse isDirty;
            # accept isClean as well for custom task metadata and test fixtures.
            checks = [
                bool(obj["isClean"])
                if "isClean" in obj
                else obj.get("isDirty") is False
                for obj in candidates
            ]
        else:
            checks = [
                obj.get(condition.state_key) == condition.expected for obj in candidates
            ]
        return all(checks) if condition.quantifier == "all" else any(checks)


def evaluate_task(
    global_goal: str, current_metadata: Mapping[str, Any]
) -> tuple[bool, float]:
    """Stateless template function requested by the experiment interface."""

    return TaskEvaluator().evaluate_task(global_goal, current_metadata)
