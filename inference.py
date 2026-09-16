"""Stateless RGB inference transport for the vanilla agent."""

from __future__ import annotations

import base64
import io
import json
import platform
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image


class APIBackend:
    """Client for nexus-model-server /generate, without automatic retries."""

    def __init__(self, endpoint: str, *, timeout: float = 120, token: str = ""):
        self.base_url = endpoint.rstrip("/")
        self.endpoint = self.base_url + "/generate"
        self.timeout = timeout
        self.token = token
        self.last_generation = {}
        self._server_info = None

    def generate_images(self, images, prompt):
        encoded = []
        for image in images:
            buffer = io.BytesIO()
            Image.fromarray(image.astype("uint8")).save(buffer, format="PNG")
            encoded.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(
            self.endpoint,
            data=json.dumps({"images": encoded, "prompt": prompt}).encode(),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise ValueError("Inference response must contain text")
        tokens = payload.get("tokens")
        if tokens is not None and (type(tokens) is not int or tokens < 0):
            raise ValueError("Invalid inference token count")
        for key in ("input_tokens", "output_tokens"):
            value = payload.get(key)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"Invalid {key}")
        finish_reason = payload.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise ValueError("Invalid finish_reason")
        self.last_generation = {
            key: payload[key]
            for key in ("tokens", "input_tokens", "output_tokens", "finish_reason")
            if key in payload
        }
        return payload["text"], tokens

    def generate(self, image, prompt):
        return self.generate_images([image], prompt)

    def health(self, *, refresh=False):
        """Return deployment metadata without triggering model generation."""

        if self._server_info is not None and not refresh:
            return dict(self._server_info)
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(self.base_url + "/health", headers=headers, method="GET")
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        if not isinstance(payload, dict) or type(payload.get("ready")) is not bool:
            raise ValueError("Invalid health response")
        self._server_info = payload
        return dict(payload)


def _backend_health(backend):
    provider = getattr(backend, "health", None)
    info = (
        provider()
        if callable(provider)
        else {
            "ready": True,
            "backend_class": type(backend).__name__,
            "model_id": getattr(backend, "model_id", "unknown"),
        }
    )
    if not isinstance(info, dict):
        raise TypeError("backend health() must return a dictionary")
    return {
        "ready": bool(info.get("ready", True)),
        "server": "nexus-model-server",
        "python": platform.python_version(),
        **info,
    }


def _normalize_generation(result):
    """Normalize legacy tuple backends and metadata-rich server backends."""

    if isinstance(result, tuple) and len(result) == 2:
        text, tokens = result
        return {"text": text, "tokens": tokens}
    if isinstance(result, dict):
        payload = dict(result)
        if not isinstance(payload.get("text"), str):
            raise TypeError("generation result must contain text")
        return payload
    raise TypeError("backend must return (text, tokens) or a result dictionary")


def make_server(backend, host="127.0.0.1", port=8001, token=""):
    """Create a server with a shared backend and no conversation storage."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "NexusModelServer/2"

        def _authorized(self):
            return not token or self.headers.get("Authorization") == f"Bearer {token}"

        def _json(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                self.send_error(404)
                return
            if not self._authorized():
                self.send_error(401)
                return
            try:
                self._json(200, _backend_health(backend))
            except Exception as exc:
                self._json(
                    503,
                    {
                        "ready": False,
                        "server": "nexus-model-server",
                        "error_type": type(exc).__name__,
                    },
                )

        def do_POST(self):
            if self.path != "/generate":
                self.send_error(404)
                return
            if not self._authorized():
                self.send_error(401)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32 * 1024 * 1024:
                    raise ValueError("Invalid request length")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Expected object")
                encoded = payload.get("images")
                prompt = payload.get("prompt")
                if (
                    not isinstance(prompt, str)
                    or not isinstance(encoded, list)
                    or not 1 <= len(encoded) <= 16
                ):
                    raise ValueError("Expected prompt and 1..16 images")
                images = []
                for item in encoded:
                    with Image.open(
                        io.BytesIO(base64.b64decode(item, validate=True))
                    ) as im:
                        if im.width * im.height > 4_000_000:
                            raise ValueError("Image too large")
                        images.append(np.array(im.convert("RGB")))
            except (ValueError, TypeError, OSError):
                self.send_error(400, "Invalid inference request")
                return
            try:
                payload = _normalize_generation(backend.generate_images(images, prompt))
                body = json.dumps(payload).encode()
            except Exception as exc:
                self._json(
                    500,
                    {
                        "error": "inference_failed",
                        "error_type": type(exc).__name__,
                    },
                )
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer((host, port), Handler)
