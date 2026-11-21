from __future__ import annotations

import unittest

from perception_service.contract import FrameEnvelope
from perception_service.navigation import build_navigation
from test_contract import valid_metadata


class NavigationTests(unittest.TestCase):
    def test_floor_and_wall_become_traversable_and_blocked(self) -> None:
        metadata = valid_metadata()
        metadata["navigation"] = {
            "source": "unity_raycast",
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
                },
                {
                    "pixel_uv": [0.55, 0.5],
                    "direction_camera": [0.02, 0, 1],
                    "hit": True,
                    "hit_distance_m": 3.0,
                    "world_position": [0.06, 1.4, 3],
                    "world_normal": [0, 0, -1],
                    "surface": "obstacle",
                },
                {
                    "pixel_uv": [0.6, 0.4],
                    "direction_camera": [0.1, 0.05, 1],
                    "hit": False,
                    "surface": "unknown",
                },
            ],
        }
        envelope = FrameEnvelope.from_dict(metadata)
        result = build_navigation(envelope.camera, envelope.navigation)
        self.assertIsNotNone(result)
        self.assertGreater(result["summary"]["traversable_cell_count"], 0)
        self.assertGreater(result["summary"]["blocked_cell_count"], 0)
        self.assertEqual(len(result["directions"]), 9)

    def test_returns_none_without_navigation(self) -> None:
        envelope = FrameEnvelope.from_dict(valid_metadata())
        self.assertIsNone(build_navigation(envelope.camera, envelope.navigation))


if __name__ == "__main__":
    unittest.main()
