"""Stateless RGB inference transport for Gemma-4-12B-it (4-bit BitsAndBytes)."""

from __future__ import annotations

import os
import threading

import numpy as np
from PIL import Image

from inference import make_server

DEFAULT_MODEL_ID = "google/gemma-4-12B-it"


class Gemma4QuantizedVLMBackend:
    """Hugging Face Transformers adapter for Gemma 4 using 4-bit quantization."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device_map: str = "auto",
        revision: str | None = None,
    ) -> None:
        try:
            import bitsandbytes  # noqa: F401
            import torch
            from transformers import (
                AutoModelForImageTextToText,
                AutoProcessor,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "필요한 패키지가 설치되지 않았습니다. 다음 명령어로 설치해주세요:\n"
                "  uv pip install bitsandbytes accelerate"
            ) from exc

        self.torch = torch
        self.model_id = model_id
        self.requested_revision = revision
        self.decoding = {"max_new_tokens": 512, "do_sample": False}

        print(f"[{model_id}] 4-bit BitsAndBytes 설정 초기화 중...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        print(f"[{model_id}] Processor 로딩 중...")
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)

        print(f"[{model_id}] 모델 가중치 4-bit 로딩 중 (device_map={device_map})...")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            revision=revision,
            quantization_config=bnb_config,
            device_map=device_map,
        ).eval()

        print(f"[{model_id}] 모델 로딩 완료!")
        self._generation_lock = threading.Lock()

    def generate(self, image: np.ndarray, prompt: str):
        return self.generate_images([image], prompt)

    def generate_images(self, images: list[np.ndarray], prompt: str):
        messages = [
            {
                "role": "user",
                "content": [
                    *(
                        {
                            "type": "image",
                            "image": Image.fromarray(image.astype("uint8")),
                        }
                        for image in images
                    ),
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        for key, value in inputs.items():
            if hasattr(value, "is_floating_point") and value.is_floating_point():
                inputs[key] = value.to(self.model.device, dtype=self.torch.bfloat16)
            elif hasattr(value, "to"):
                inputs[key] = value.to(self.model.device)

        input_length = int(inputs["input_ids"].shape[-1])
        with self._generation_lock, self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                **self.decoding,
            )

        generated = output[0][input_length:]
        text = self.processor.decode(generated, skip_special_tokens=True)
        output_tokens = int(generated.shape[-1])
        eos_id = getattr(self.model.generation_config, "eos_token_id", None)
        last_token = int(generated[-1]) if output_tokens else None
        eos_ids = {eos_id} if isinstance(eos_id, int) else set(eos_id or [])
        return {
            "text": text,
            "tokens": input_length + output_tokens,
            "input_tokens": input_length,
            "output_tokens": output_tokens,
            "finish_reason": "eos" if last_token in eos_ids else "length",
        }

    def health(self):
        import bitsandbytes
        import transformers

        config = getattr(self.model, "config", None)
        revision = getattr(config, "_commit_hash", None) or self.requested_revision
        gpu = []
        if self.torch.cuda.is_available():
            gpu = [
                self.torch.cuda.get_device_name(index)
                for index in range(self.torch.cuda.device_count())
            ]
        return {
            "ready": True,
            "model_id": self.model_id,
            "model_revision": revision or "unknown",
            "processor_class": type(self.processor).__name__,
            "model_class": type(self.model).__name__,
            "quantization": "NF4 4-bit",
            "decoding": dict(self.decoding),
            "libraries": {
                "torch": self.torch.__version__,
                "transformers": transformers.__version__,
                "bitsandbytes": bitsandbytes.__version__,
            },
            "gpu": gpu,
        }


def main():
    model_id = os.getenv("NEXUS_VLM_MODEL", DEFAULT_MODEL_ID)
    revision = os.getenv("NEXUS_VLM_REVISION") or None
    backend = Gemma4QuantizedVLMBackend(model_id=model_id, revision=revision)

    host = os.getenv("NEXUS_MODEL_HOST", "127.0.0.1")
    port = int(os.getenv("NEXUS_MODEL_PORT", "8001"))
    token = os.getenv("NEXUS_API_TOKEN", "")

    print(f"Inference HTTP 서버 실행: http://{host}:{port}/generate")
    server = make_server(backend, host, port, token)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
