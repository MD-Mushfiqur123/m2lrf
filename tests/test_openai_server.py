"""
Unit tests for M-2LRF OpenAI-Compatible Serving Server.
"""

import json
import socket
import time
import unittest
import urllib.request
import urllib.error

import torch
import torch.nn as nn

from m2lrf.serving.engine import LLMEngine
from m2lrf.serving.openai_server import OpenAIServer, OpenAIServingHandler


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestOpenAIServer(unittest.TestCase):
    """Tests OpenAI API compliance and serving functionality."""

    @classmethod
    def setUpClass(cls):
        cls.port = get_free_port()
        vocab_size = 256
        hidden_dim = 64

        # Lightweight forward function for testing
        linear = nn.Linear(hidden_dim, vocab_size, bias=False)
        embed = nn.Embedding(vocab_size, hidden_dim)

        def mock_forward(input_ids, kv_cache, block_table):
            h = embed(input_ids)
            logits = linear(h)
            return logits

        cls.engine = LLMEngine(
            model_forward_fn=mock_forward,
            num_gpu_blocks=64,
            block_size=16,
            num_kv_heads=4,
            head_dim=16,
            device="cpu",
        )

        cls.server = OpenAIServer(
            engine=cls.engine,
            model_name="m2lrf-test-2bit",
            host="127.0.0.1",
            port=cls.port,
        )
        cls.server.start(block=False)
        time.sleep(0.3)  # Allow socket to bind

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def _http_get(self, endpoint: str) -> dict:
        url = f"http://127.0.0.1:{self.port}{endpoint}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as response:
            self.assertEqual(response.status, 200)
            return json.loads(response.read().decode("utf-8"))

    def _http_post(self, endpoint: str, body: dict) -> dict:
        url = f"http://127.0.0.1:{self.port}{endpoint}"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as response:
            self.assertEqual(response.status, 200)
            return json.loads(response.read().decode("utf-8"))

    def test_01_health_endpoint(self):
        res = self._http_get("/health")
        self.assertEqual(res.get("status"), "healthy")
        self.assertIn("M-2LRF", res.get("serving_engine", ""))

    def test_02_models_endpoint(self):
        res = self._http_get("/v1/models")
        self.assertEqual(res.get("object"), "list")
        data = res.get("data", [])
        self.assertGreater(len(data), 0)
        self.assertEqual(data[0]["id"], "m2lrf-test-2bit")

    def test_03_metrics_endpoint(self):
        res = self._http_get("/metrics")
        self.assertIn("throughput_tokens_per_sec", res)
        self.assertIn("kv_cache_utilization", res)
        self.assertIn("total_tokens_generated", res)

    def test_04_chat_completions_sync(self):
        body = {
            "model": "m2lrf-test-2bit",
            "messages": [
                {"role": "user", "content": "What is 2+2?"}
            ],
            "max_tokens": 8,
            "temperature": 0.0,
            "stream": False,
        }
        res = self._http_post("/v1/chat/completions", body)
        self.assertEqual(res.get("object"), "chat.completion")
        self.assertIn("choices", res)
        self.assertGreater(len(res["choices"]), 0)
        choice = res["choices"][0]
        self.assertIn("message", choice)
        self.assertEqual(choice["message"]["role"], "assistant")
        self.assertIn("usage", res)
        self.assertGreater(res["usage"]["total_tokens"], 0)

    def test_05_chat_completions_streaming(self):
        url = f"http://127.0.0.1:{self.port}/v1/chat/completions"
        body = {
            "model": "m2lrf-test-2bit",
            "messages": [
                {"role": "user", "content": "Tell me a short reason"}
            ],
            "max_tokens": 6,
            "temperature": 0.0,
            "stream": True,
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        chunks = []
        with urllib.request.urlopen(req, timeout=10) as response:
            self.assertEqual(response.status, 200)
            for line in response:
                line_str = line.decode("utf-8").strip()
                if line_str.startswith("data: "):
                    payload_str = line_str[6:]
                    if payload_str == "[DONE]":
                        chunks.append("[DONE]")
                        break
                    else:
                        chunks.append(json.loads(payload_str))


        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[-1], "[DONE]")
        first_chunk = chunks[0]
        self.assertEqual(first_chunk.get("object"), "chat.completion.chunk")

    def test_06_raw_completions(self):
        body = {
            "model": "m2lrf-test-2bit",
            "prompt": "Solve: x + 5 = 12",
            "max_tokens": 5,
            "temperature": 0.0,
        }
        res = self._http_post("/v1/completions", body)
        self.assertEqual(res.get("object"), "text_completion")
        self.assertIn("choices", res)
        self.assertGreater(len(res["choices"]), 0)


if __name__ == "__main__":
    unittest.main()
