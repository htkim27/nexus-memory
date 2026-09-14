"""Vision-language high-level policy and per-step cost instrumentation."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np

DEFAULT_VLM_MODEL = "google/gemma-3-4b-it"


class VLMBackend(Protocol):
    """Minimal adapter contract for local or hosted vision-language models."""

    def generate(self, image: np.ndarray, prompt: str) -> tuple[str, int | None]:
        """Return generated text and, when known, total input+output tokens."""


@dataclass(frozen=True)
class StepMetrics:
    step: int
    tokens: int
    latency_seconds: float


class VLMAgent:
    """High-level policy that emits a memory rewrite and one textual subtask."""

    SYSTEM_PROMPT = """You are the high-level policy for an embodied agent.
You see only the current RGB frame, never simulator metadata. Use the memory to
avoid repeating failed actions. Choose exactly one small, executable subtask.
Return JSON only: {"new_memory": "concise updated memory", "subtask": "..."}.

SUBTASK CONTRACT:
- Keep subtask as a natural-language string, not a nested object or action JSON.
- Use exactly one of these canonical forms:
  "Move ahead", "Move back", "Turn left", "Turn right", "Look up", "Look down",
  "Open the <object>", "Close the <object>", "Pick up the <object>",
  "Put the held object in/on the <receptacle>", "Turn on/off the <object>",
  "Slice the <object>", "Break the <object>", "Clean the <object>",
  "Dirty the <object>", "Fill the <object> with water/coffee/wine",
  "Empty the <object>", or "Stop".
- For manipulation, name a target visible in the RGB image using its ordinary
  object type. Never emit simulator object IDs, coordinates, or metadata.
- A bare noun such as "Fridge" or "Fridge door" is invalid. Say "Open the fridge".
- If a prior action failed, change the plan or first satisfy its precondition.
"""

    def __init__(
        self, backend: VLMBackend, logger: logging.Logger | None = None
    ) -> None:
        self.backend = backend
        self.logger = logger or logging.getLogger("nexus_memory.vlm")
        self.metrics: list[StepMetrics] = []

    def reset_metrics(self) -> None:
        self.metrics.clear()

    def step(
        self, rgb_image: np.ndarray, memory_context: str, global_goal: str
    ) -> tuple[str, str]:
        """Run one VLM decision using only RGB, memory, and the global goal."""

        if not isinstance(rgb_image, np.ndarray) or rgb_image.ndim != 3:
            raise ValueError("rgb_image must be an HxWxC numpy array")
        prompt = (
            f"{self.SYSTEM_PROMPT}\n\nGLOBAL GOAL:\n{global_goal}\n\n"
            f"MEMORY:\n{memory_context or '(empty)'}"
        )
        started = time.perf_counter()
        raw_text, backend_tokens = self.backend.generate(rgb_image, prompt)
        latency = time.perf_counter() - started
        payload = self._parse_json(raw_text)
        tokens = (
            backend_tokens
            if backend_tokens is not None
            else self._estimate_tokens(prompt + raw_text)
        )
        metric = StepMetrics(len(self.metrics) + 1, int(tokens), latency)
        self.metrics.append(metric)
        self.logger.info("vlm_step %s", asdict(metric))
        return str(payload["new_memory"]), str(payload["subtask"])

    def cost_summary(self) -> str:
        total_tokens = sum(item.tokens for item in self.metrics)
        total_latency = sum(item.latency_seconds for item in self.metrics)
        return (
            f"steps={len(self.metrics)} | tokens={total_tokens} | "
            f"latency={total_latency:.2f}s"
        )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise ValueError(
                    f"VLM did not return a JSON object: {text[:200]!r}"
                ) from None
            payload = json.loads(match.group(0))
        if not isinstance(payload, dict) or not all(
            isinstance(payload.get(key), str) for key in ("new_memory", "subtask")
        ):
            raise ValueError(
                "VLM JSON must contain string new_memory and subtask fields"
            )
        return payload

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Portable fallback only. Production adapters should return tokenizer counts.
        return max(1, len(text) // 4)


class Gemma3VLMBackend:
    """Hugging Face Transformers adapter for multimodal Gemma 3 models."""

    def __init__(
        self,
        model_id: str = DEFAULT_VLM_MODEL,
        *,
        device_map: str = "auto",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:  # pragma: no cover - optional heavyweight deps
            raise RuntimeError(
                "Install `torch transformers accelerate` to use this backend."
            ) from exc
        self.torch = torch
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map=device_map,
            dtype=torch.bfloat16,
        ).eval()
        # A model instance is shared across dashboard sessions. Serialize
        # generation to avoid concurrent writes to the same CUDA KV cache.
        self._generation_lock = threading.Lock()

    def generate(self, image: np.ndarray, prompt: str) -> tuple[str, int]:
        from PIL import Image

        messages = self._build_messages(Image.fromarray(image.astype("uint8")), prompt)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Keep token IDs integer while moving floating image tensors to BF16.
        for key, value in inputs.items():
            if hasattr(value, "is_floating_point") and value.is_floating_point():
                inputs[key] = value.to(self.model.device, dtype=self.model.dtype)
            elif hasattr(value, "to"):
                inputs[key] = value.to(self.model.device)
        input_length = int(inputs["input_ids"].shape[-1])
        with self._generation_lock, self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        generated = output[0][input_length:]
        text = self.processor.decode(generated, skip_special_tokens=True)
        return text, int(output.shape[-1])

    @staticmethod
    def _build_messages(image: Any, prompt: str) -> list[dict[str, Any]]:
        """Build Gemma 3's native multimodal chat message structure."""

        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]


class DemoVLMBackend:
    """Dependency-free UI smoke-test backend; not suitable for experiments."""

    def generate(self, image: np.ndarray, prompt: str) -> tuple[str, int | None]:
        del image
        return json.dumps(
            {
                "new_memory": "Demo policy active; inspect the current room.",
                "subtask": "Turn right",
            }
        ), None
