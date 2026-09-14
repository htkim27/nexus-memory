"""Dependency-light contract tests for the foundational components."""

from __future__ import annotations

import unittest

from agent import Gemma3VLMBackend, VLMAgent
from app import ResearchSession, SessionStore
from evaluator import GoalCondition, TaskEvaluator
from memory import HierarchicalMemory, MemFlatMemory
from translator import ActionTranslator, TranslationError

try:
    import numpy as np

    from environment import ThorEnvironment
except (
    ImportError
):  # Allows pure-logic tests before runtime dependencies are installed.
    np = None
    ThorEnvironment = None  # type: ignore[assignment,misc]


class FakeEvent:
    def __init__(self, success: bool = True, error: str = "") -> None:
        self.frame = np.zeros((4, 5, 3), dtype=np.uint8)
        self.metadata = {
            "lastActionSuccess": success,
            "errorMessage": error,
            "objects": [
                {"objectId": "Apple|1", "objectType": "Apple", "visible": True}
            ],
        }


class FakeController:
    def __init__(self) -> None:
        self.last_event = FakeEvent()
        self.command = None

    def reset(self, scene: str) -> FakeEvent:
        self.last_event = FakeEvent()
        return self.last_event

    def step(self, **command: object) -> FakeEvent:
        self.command = command
        self.last_event = FakeEvent(False, "physical precondition failed")
        return self.last_event

    def stop(self) -> None:
        pass


@unittest.skipIf(np is None, "numpy runtime dependency is not installed")
class EnvironmentTests(unittest.TestCase):
    def test_public_result_excludes_metadata_and_preserves_feedback(self) -> None:
        controller = FakeController()
        environment = ThorEnvironment(controller=controller)
        result = environment.execute({"action": "SliceObject", "objectId": "Apple|1"})
        self.assertFalse(hasattr(result, "metadata"))
        self.assertEqual(result.rgb.shape, (4, 5, 3))
        self.assertFalse(result.feedback.lastActionSuccess)
        self.assertEqual(result.feedback.errorMessage, "physical precondition failed")


class TranslatorTests(unittest.TestCase):
    def test_object_action_is_grounded_to_supplied_id(self) -> None:
        command = ActionTranslator().translate(
            "Open the fridge", ["Fridge|1", "Apple|2"]
        )
        self.assertEqual(command, {"action": "OpenObject", "objectId": "Fridge|1"})

    def test_unknown_object_fails_closed(self) -> None:
        with self.assertRaises(TranslationError):
            ActionTranslator().translate("Slice the tomato", ["Apple|2"])

    def test_llm_mapper_output_is_strictly_validated(self) -> None:
        translator = ActionTranslator(
            lambda _text, _ids: {"action": "RotateRight", "objectId": "Apple|2"}
        )
        with self.assertRaises(TranslationError):
            translator.translate("Turn right", ["Apple|2"])


class AgentTests(unittest.TestCase):
    def test_gemma3_messages_contain_one_image_and_policy_prompt(self) -> None:
        image = object()
        messages = Gemma3VLMBackend._build_messages(image, "policy prompt")
        self.assertEqual(messages[0]["role"], "user")
        self.assertIs(messages[0]["content"][0]["image"], image)
        self.assertEqual(messages[0]["content"][1]["text"], "policy prompt")

    def test_prompt_requires_canonical_natural_language_subtasks(self) -> None:
        self.assertIn('"Open the <object>"', VLMAgent.SYSTEM_PROMPT)
        self.assertIn('bare noun such as "Fridge"', VLMAgent.SYSTEM_PROMPT)
        self.assertIn("not a nested object", VLMAgent.SYSTEM_PROMPT)


class MemoryTests(unittest.TestCase):
    def test_failures_are_forced_into_both_memory_variants(self) -> None:
        for memory in (MemFlatMemory(), HierarchicalMemory()):
            memory.reset("Slice the apple")
            context = memory.update_memory(
                "Knife is on counter",
                "Slice the apple",
                {"action": "SliceObject", "objectId": "Apple|2"},
                False,
                "Cannot slice without holding a knife",
            )
            self.assertIn("FAILURE", context)
            self.assertIn("Cannot slice without holding a knife", context)
            self.assertNotIn("Apple|2", context)
            self.assertIn('"targetType": "Apple"', context)

    def test_feedback_object_ids_are_removed_from_vlm_memory(self) -> None:
        memory = MemFlatMemory()
        context = memory.update_memory(
            "I previously saw Apple|1|2|3",
            "Pick up the apple",
            {"action": "PickupObject", "objectId": "Apple|1|2|3"},
            False,
            "Apple|1|2|3 is too far from Knife|4|5|6",
        )
        self.assertNotIn("Apple|1|2|3", context)
        self.assertNotIn("Knife|4|5|6", context)
        self.assertIn("I previously saw Apple", context)
        self.assertIn("Apple is too far from Knife", context)


class ClosingEnvironment:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class SessionTests(unittest.TestCase):
    @staticmethod
    def _session(environment: ClosingEnvironment) -> ResearchSession:
        return ResearchSession(
            environment=environment,  # type: ignore[arg-type]
            memory=MemFlatMemory(),
            agent=None,  # type: ignore[arg-type]
            translator=ActionTranslator(),
            evaluator=TaskEvaluator(),
            global_goal="test",
        )

    def test_remove_closes_environment_exactly_once(self) -> None:
        store = SessionStore()
        environment = ClosingEnvironment()
        session_id = store.replace(None, self._session(environment))
        self.assertTrue(store.remove(session_id))
        self.assertFalse(store.remove(session_id))
        self.assertEqual(environment.close_calls, 1)

    def test_expired_session_is_closed(self) -> None:
        store = SessionStore()
        environment = ClosingEnvironment()
        session = self._session(environment)
        session.last_accessed = 0
        store.replace(None, session)
        self.assertEqual(store.cleanup_expired(), 1)
        self.assertEqual(environment.close_calls, 1)


class EvaluatorTests(unittest.TestCase):
    def test_compound_goal_and_score(self) -> None:
        metadata = {
            "objects": [
                {"objectType": "Plate", "isDirty": False},
                {"objectType": "Plate", "isDirty": False},
                {"objectType": "Apple", "isSliced": False},
            ]
        }
        evaluator = TaskEvaluator()
        success, score = evaluator.evaluate_task(
            "Clean all plates and slice the apple", metadata
        )
        self.assertFalse(success)
        self.assertEqual(score, 0.5)

    def test_explicit_registry_is_supported(self) -> None:
        goal = "benchmark-task-001"
        evaluator = TaskEvaluator(
            {goal: [GoalCondition("Apple", "isSliced", True, "all")]}
        )
        self.assertEqual(
            evaluator.evaluate_task(
                goal, {"objects": [{"objectType": "Apple", "isSliced": True}]}
            ),
            (True, 1.0),
        )

    def test_custom_is_clean_metadata_is_supported(self) -> None:
        result = TaskEvaluator().evaluate_task(
            "Clean all plates",
            {"objects": [{"objectType": "Plate", "isClean": True}]},
        )
        self.assertEqual(result, (True, 1.0))

    def test_placement_uses_parent_receptacles(self) -> None:
        metadata = {
            "objects": [
                {
                    "objectType": "Apple",
                    "parentReceptacles": ["Fridge|1"],
                }
            ]
        }
        self.assertEqual(
            TaskEvaluator().evaluate_task("Put the apple in the fridge", metadata),
            (True, 1.0),
        )


if __name__ == "__main__":
    unittest.main()
