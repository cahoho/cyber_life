from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from perception_service.backends import HeuristicBackend
from perception_service.contract import FrameEnvelope
from perception_service.engine import VisionEngine
from perception_service.semantics import SemanticAnalyzer, SemanticSettings
from test_contract import valid_metadata


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class SemanticAnalyzerTests(unittest.TestCase):
    def test_responses_request_contains_image_and_json_schema(self) -> None:
        image = np.full((360, 640, 3), (210, 205, 195), dtype=np.uint8)
        frame = FrameEnvelope.from_dict(valid_metadata())
        snapshot = VisionEngine(HeuristicBackend()).process(image, frame)
        _, jpeg = cv2.imencode(".jpg", image)
        answer_text = json.dumps(
            {
                "answer": "当前看不到明确物体。",
                "grounded": True,
                "confidence": 0.6,
                "evidence": ["画面"],
            },
            ensure_ascii=False,
        )
        response = _FakeResponse(
            {
                "output": [
                    {
                        "content": [
                            {"type": "output_text", "text": answer_text}
                        ]
                    }
                ]
            }
        )
        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return response

        analyzer = SemanticAnalyzer(
            SemanticSettings(
                api_key="test-key",
                model="deepseek-flash",
                base_url="https://api.deepseek.com",
                provider="deepseek",
            )
        )
        with patch(
            "perception_service.semantics.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            result = analyzer.answer_question(
                "场景里有什么？",
                jpeg.tobytes(),
                snapshot,
                None,
                None,
            )

        self.assertEqual(result["source"], "vlm")
        self.assertEqual(result["answer"], "当前看不到明确物体。")
        request = captured["request"]
        self.assertEqual(request.full_url, "https://api.deepseek.com/responses")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        payload = json.loads(request.data.decode("utf-8"))
        content = payload["input"][0]["content"]
        self.assertEqual(content[1]["type"], "input_image")
        self.assertTrue(content[1]["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(payload["text"]["format"]["type"], "json_schema")
        self.assertTrue(payload["text"]["format"]["strict"])
        self.assertEqual(payload["reasoning"], {"effort": "none"})
        self.assertFalse(payload["store"])


if __name__ == "__main__":
    unittest.main()
