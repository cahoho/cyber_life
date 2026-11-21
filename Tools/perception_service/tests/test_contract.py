from __future__ import annotations

import unittest

from perception_service.contract import ContractError, FrameEnvelope


def valid_metadata() -> dict:
    return {
        "schema_version": "1.0",
        "session_id": "sess_test",
        "frame_id": 7,
        "capture_ts_ms": 1_700_000_000_000,
        "camera": {
            "width": 640,
            "height": 360,
            "fov_y_deg": 65,
            "near": 0.1,
            "far": 100,
            "position": [0, 1.4, 0],
            "rotation_xyzw": [0, 0, 0, 1],
            "unity_to_cv": True,
        },
        "request": {"fast": True, "slow": False},
    }


def valid_navigation() -> dict:
    return {
        "source": "unity_raycast",
        "ground_y": 0.0,
        "max_slope_deg": 45,
        "step_height_m": 0.2,
        "rays": [
            {
                "pixel_uv": [0.5, 0.75],
                "direction_camera": [0, -0.2, 0.98],
                "hit": True,
                "hit_distance_m": 2.0,
                "world_position": [0, 0, 2],
                "world_normal": [0, 1, 0],
                "surface": "floor",
            }
        ],
    }


class FrameEnvelopeTests(unittest.TestCase):
    def test_valid_frame(self) -> None:
        envelope = FrameEnvelope.from_dict(valid_metadata())
        self.assertEqual(envelope.frame_id, 7)
        self.assertEqual(envelope.camera.width, 640)
        self.assertTrue(envelope.request_fast)
        self.assertIsNone(envelope.navigation)

    def test_accepts_navigation_rays(self) -> None:
        metadata = valid_metadata()
        metadata["navigation"] = valid_navigation()
        envelope = FrameEnvelope.from_dict(metadata)
        self.assertIsNotNone(envelope.navigation)
        self.assertEqual(envelope.navigation.rays[0].surface, "floor")

    def test_rejects_wrong_schema(self) -> None:
        metadata = valid_metadata()
        metadata["schema_version"] = "2.0"
        with self.assertRaises(ContractError):
            FrameEnvelope.from_dict(metadata)

    def test_rejects_invalid_quaternion(self) -> None:
        metadata = valid_metadata()
        metadata["camera"]["rotation_xyzw"] = [0, 0, 0, 0]
        with self.assertRaises(ContractError):
            FrameEnvelope.from_dict(metadata)

    def test_rejects_malformed_navigation_ray(self) -> None:
        metadata = valid_metadata()
        metadata["navigation"] = valid_navigation()
        metadata["navigation"]["rays"][0]["direction_camera"] = [0, 0, 0]
        with self.assertRaises(ContractError):
            FrameEnvelope.from_dict(metadata)


if __name__ == "__main__":
    unittest.main()
