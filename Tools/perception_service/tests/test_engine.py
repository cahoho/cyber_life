from __future__ import annotations

import unittest

import cv2
import numpy as np

from perception_service.backends import HeuristicBackend
from perception_service.contract import Detection, FrameEnvelope
from perception_service.engine import VisionEngine
from test_contract import valid_metadata


class EngineTests(unittest.TestCase):
    def test_snapshot_contract_without_model(self) -> None:
        metadata = valid_metadata()
        frame = FrameEnvelope.from_dict(metadata)
        image = np.full((360, 640, 3), (220, 210, 180), dtype=np.uint8)
        cv2.rectangle(image, (280, 130), (370, 300), (20, 110, 235), -1)
        snapshot = VisionEngine(HeuristicBackend()).process(image, frame)
        payload = snapshot.to_dict()
        self.assertEqual(payload["source_frame_id"], 7)
        self.assertIn("scene", payload)
        self.assertIn("vision_events", payload)
        for item in payload["objects"]:
            self.assertIsNone(item["world_position"])
            self.assertEqual(item["depth_source"], "none")
            self.assertGreater(item["track_id"], 0)

    def test_ray_hit_adds_object_depth_and_world_position(self) -> None:
        class FixedBackend:
            name = "fixed"

            @property
            def ready(self) -> bool:
                return True

            def process(self, image, envelope):
                return [
                    Detection(
                        label="box",
                        confidence=0.9,
                        bbox_xyxy=(280, 130, 370, 300),
                        track_id=4,
                    )
                ]

            def close(self) -> None:
                return None

        metadata = valid_metadata()
        metadata["navigation"] = {
            "ground_y": 0.0,
            "rays": [
                {
                    "pixel_uv": [0.5, 0.638],
                    "direction_camera": [0, -0.05, 0.9987],
                    "hit": True,
                    "hit_distance_m": 2.1,
                    "world_position": [0, 0.8, 2.1],
                    "world_normal": [0, 0, -1],
                    "surface": "obstacle",
                }
            ],
        }
        frame = FrameEnvelope.from_dict(metadata)
        image = np.full((360, 640, 3), (220, 210, 180), dtype=np.uint8)
        snapshot = VisionEngine(FixedBackend()).process(image, frame)
        self.assertTrue(snapshot.objects)
        positioned = [
            item for item in snapshot.objects if item["depth_source"] == "unity_raycast"
        ]
        self.assertTrue(positioned)
        self.assertGreater(positioned[0]["position_confidence"], 0)


if __name__ == "__main__":
    unittest.main()
