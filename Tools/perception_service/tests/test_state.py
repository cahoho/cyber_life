from __future__ import annotations

import unittest

import cv2
import numpy as np

from perception_service.backends import HeuristicBackend
from perception_service.contract import FrameEnvelope
from perception_service.engine import VisionEngine
from perception_service.semantics import SemanticAnalyzer, SemanticSettings
from perception_service.state import SceneMemory
from test_contract import valid_metadata


class SceneMemoryTests(unittest.TestCase):
    def test_answers_object_question_without_api(self) -> None:
        image = np.full((360, 640, 3), (220, 210, 180), dtype=np.uint8)
        cv2.rectangle(image, (280, 130), (370, 300), (20, 110, 235), -1)
        envelope = FrameEnvelope.from_dict(valid_metadata())
        snapshot = VisionEngine(HeuristicBackend()).process(image, envelope)
        memory = SceneMemory(
            SemanticAnalyzer(
                SemanticSettings(api_key="", model="none", base_url="http://unused")
            )
        )
        _, jpeg = cv2.imencode(".jpg", image)
        memory.update(envelope, jpeg.tobytes(), snapshot, snapshot.navigation)
        answer = memory.ask("当前场景里有什么？")
        self.assertEqual(answer["source"], "fast_state")
        self.assertTrue(answer["answer"])

    def test_answers_navigation_question(self) -> None:
        metadata = valid_metadata()
        metadata["navigation"] = {
            "ground_y": 0.0,
            "rays": [
                {
                    "pixel_uv": [0.5, 0.8],
                    "direction_camera": [0, -0.2, 0.98],
                    "hit": True,
                    "hit_distance_m": 2.0,
                    "world_position": [0, 0, 2],
                    "world_normal": [0, 1, 0],
                    "surface": "floor",
                }
            ],
        }
        image = np.full((360, 640, 3), (220, 210, 180), dtype=np.uint8)
        envelope = FrameEnvelope.from_dict(metadata)
        snapshot = VisionEngine(HeuristicBackend()).process(image, envelope)
        memory = SceneMemory(
            SemanticAnalyzer(
                SemanticSettings(api_key="", model="none", base_url="http://unused")
            )
        )
        _, jpeg = cv2.imencode(".jpg", image)
        memory.update(envelope, jpeg.tobytes(), snapshot, snapshot.navigation)
        answer = memory.ask("前方能通行吗？")
        self.assertIn("净空", answer["answer"])


if __name__ == "__main__":
    unittest.main()
