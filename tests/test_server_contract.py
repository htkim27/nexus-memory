"""HTTP contract for the vanilla agent's model server."""

from __future__ import annotations

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

import inference
import inference_2


class ContractBackend:
    model_id = "contract/fake"

    def __init__(self):
        self.calls = []

    def health(self):
        return {
            "ready": True,
            "model_id": self.model_id,
            "model_revision": "fixture-revision",
            "model_class": type(self).__name__,
            "processor_class": "FakeProcessor",
            "quantization": "none",
            "decoding": {"max_new_tokens": 512, "do_sample": False},
            "libraries": {"fake": "1.0"},
            "gpu": [],
        }

    def generate_images(self, images, prompt):
        self.calls.append((images, prompt))
        return {
            "text": json.dumps([int(image[0, 0, 0]) for image in images]),
            "tokens": 15,
            "input_tokens": 11,
            "output_tokens": 4,
            "finish_reason": "eos",
        }


@pytest.fixture(params=[inference.make_server, inference_2.make_server])
def server_client(request):
    backend = ContractBackend()
    server = request.param(backend, port=0, token="secret")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = inference.APIBackend(
        f"http://127.0.0.1:{server.server_port}", token="secret"
    )
    try:
        yield backend, client
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_health_exposes_actual_server_configuration(server_client):
    _, client = server_client
    health = client.health()
    assert health["ready"] is True
    assert health["model_id"] == "contract/fake"
    assert health["model_revision"] == "fixture-revision"
    assert health["decoding"]["max_new_tokens"] == 512


def test_generate_preserves_image_order_and_generation_metadata(server_client):
    backend, client = server_client
    images = [np.full((3, 4, 3), value, dtype=np.uint8) for value in (3, 9, 1)]
    text, tokens = client.generate_images(images, "ordered")
    assert json.loads(text) == [3, 9, 1]
    assert tokens == 15
    assert client.last_generation == {
        "tokens": 15,
        "input_tokens": 11,
        "output_tokens": 4,
        "finish_reason": "eos",
    }
    assert backend.calls[0][1] == "ordered"


def test_requests_are_stateless_and_auth_is_applied_to_health(server_client):
    backend, client = server_client
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    client.generate_images([image], "first")
    client.generate_images([image], "second")
    assert [call[1] for call in backend.calls] == ["first", "second"]
    unauthorized = inference.APIBackend(client.base_url)
    with pytest.raises(HTTPError) as error:
        unauthorized.health()
    assert error.value.code == 401


def test_invalid_request_reports_transport_failure_without_backend_call(server_client):
    backend, client = server_client
    request = Request(
        client.endpoint,
        data=b'{"images":[],"prompt":"bad"}',
        headers={"Content-Type": "application/json", "Authorization": "Bearer secret"},
        method="POST",
    )
    with pytest.raises(HTTPError) as error:
        urlopen(request)
    assert error.value.code == 400
    assert backend.calls == []
