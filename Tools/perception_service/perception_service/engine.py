from __future__ import annotations

import math
import time
from collections import OrderedDict
from statistics import median

import numpy as np

from .backends import VisionBackend
from .contract import Detection, FrameEnvelope, Snapshot, event
from .navigation import build_navigation
from .semantics import infer_place_hint


class VisionEngine:
    def __init__(
        self,
        backend: VisionBackend,
        *,
        track_lost_ms: int = 2000,
        event_history: int = 128,
    ) -> None:
        self.backend = backend
        self._track_lost_ms = track_lost_ms
        self._event_history: OrderedDict[int, None] = OrderedDict()
        self._event_history_limit = event_history
        self._last_seen: dict[int, int] = {}
        self._lost: set[int] = set()
        self._previous_labels: set[str] = set()
        self._previous_centers: dict[int, tuple[float, float]] = {}

    def process(self, frame: np.ndarray, envelope: FrameEnvelope) -> Snapshot:
        started = time.perf_counter()
        now_ms = int(time.time() * 1000)
        if envelope.session_id != getattr(self, "_session_id", None):
            self._session_id = envelope.session_id
            self._last_seen.clear()
            self._lost.clear()
            self._previous_labels.clear()
            self._previous_centers.clear()
            self._event_history.clear()
            if hasattr(self, "_track_created"):
                self._track_created.clear()

        detections = self.backend.process(frame, envelope)
        detections = self._deduplicate(detections)
        objects = [self._object_dict(item, envelope, now_ms) for item in detections]
        navigation = build_navigation(envelope.camera, envelope.navigation)
        if navigation is not None:
            navigation["frame_id"] = envelope.frame_id
        place_hint, place_confidence = infer_place_hint(objects)

        current_ids = {int(item["track_id"]) for item in objects}
        vision_events: list[dict[str, object]] = []
        for track_id in sorted(current_ids):
            if track_id not in self._last_seen:
                vision_events.extend(
                    self._emit(
                        event("TRACK_APPEARED", envelope.capture_ts_ms, track_id),
                        "TRACK_APPEARED",
                    )
                )
            self._last_seen[track_id] = envelope.capture_ts_ms
            self._lost.discard(track_id)

        for track_id, last_seen in sorted(self._last_seen.items()):
            if (
                track_id not in current_ids
                and track_id not in self._lost
                and now_ms - last_seen >= self._track_lost_ms
            ):
                self._lost.add(track_id)
                vision_events.extend(
                    self._emit(
                        event("TARGET_LOST", envelope.capture_ts_ms, track_id),
                        "TARGET_LOST",
                    )
                )

        labels = {str(item["label"]) for item in objects}
        centers = {
            int(item["track_id"]): tuple(item["center_uv"]) for item in objects
        }
        novelty = self._compute_novelty(labels, centers)
        major_change = novelty >= 0.6
        if major_change:
            vision_events.extend(
                self._emit(event("SCENE_CHANGED", envelope.capture_ts_ms), "SCENE_CHANGED")
            )
        unknown_count = sum(1 for item in objects if item["label"] == "other")
        if unknown_count:
            for item in objects:
                if item["label"] == "other":
                    vision_events.extend(
                        self._emit(
                            event(
                                "NOVEL_OBJECT",
                                envelope.capture_ts_ms,
                                int(item["track_id"]),
                            ),
                            f"NOVEL_OBJECT:{item['track_id']}",
                        )
                    )

        vision_events.extend(
            self._emit(event("PERCEPTION_UPDATED", envelope.capture_ts_ms), "PERCEPTION_UPDATED")
        )
        self._previous_labels = labels
        self._previous_centers = centers  # type: ignore[assignment]
        infer_ms = max(0, int(round((time.perf_counter() - started) * 1000)))
        return Snapshot(
            frame=envelope,
            result_ts_ms=now_ms,
            infer_ms=infer_ms,
            objects=objects,
            novelty=novelty,
            major_change=major_change,
            unknown_object_count=unknown_count,
            vision_events=vision_events,
            backend=self.backend.name,
            navigation=navigation,
            place_hint=place_hint,
            place_confidence=place_confidence,
        )

    def _object_dict(
        self, detection: Detection, envelope: FrameEnvelope, now_ms: int
    ) -> dict[str, object]:
        width = envelope.camera.width
        height = envelope.camera.height
        x1, y1, x2, y2 = detection.bbox_xyxy
        center_u = max(0.0, min(1.0, ((x1 + x2) * 0.5) / width))
        center_v = max(0.0, min(1.0, ((y1 + y2) * 0.5) / height))
        area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(width * height)
        track_age_s = self._track_age(detection.track_id, envelope.capture_ts_ms)
        previous_center = self._previous_centers.get(detection.track_id or -1)
        moving = False
        velocity_hint = [0.0, 0.0]
        if previous_center is not None:
            velocity_hint = [
                round(center_u - previous_center[0], 4),
                round(center_v - previous_center[1], 4),
            ]
            moving = math.hypot(*velocity_hint) > 0.004
        depth_m, world_position, position_confidence, depth_source = (
            self._position_from_rays(detection, envelope)
        )
        return {
            "track_id": int(detection.track_id or 0),
            "label": detection.label,
            "label_source": detection.label_source,
            "confidence": round(detection.confidence, 4),
            "bbox_xyxy": [round(value, 2) for value in detection.bbox_xyxy],
            "center_uv": [round(center_u, 4), round(center_v, 4)],
            "bbox_area_ratio": round(area_ratio, 5),
            "depth_m": depth_m,
            "depth_source": depth_source,
            "world_position": world_position,
            "position_confidence": position_confidence,
            "moving": moving,
            "velocity_hint": velocity_hint,
            "track_age_s": round(track_age_s, 3),
            "visibility": "visible",
        }

    @staticmethod
    def _position_from_rays(
        detection: Detection,
        envelope: FrameEnvelope,
    ) -> tuple[float | None, list[float] | None, float, str]:
        observation = envelope.navigation
        if observation is None:
            return None, None, 0.0, "none"

        x1, y1, x2, y2 = detection.bbox_xyxy
        width = max(1.0, envelope.camera.width)
        height = max(1.0, envelope.camera.height)
        left = max(0.0, x1 / width - 0.03)
        right = min(1.0, x2 / width + 0.03)
        top = max(0.0, y1 / height - 0.03)
        bottom = min(1.0, y2 / height + 0.03)
        inside = [
            ray
            for ray in observation.rays
            if ray.hit
            and left <= ray.pixel_uv[0] <= right
            and top <= ray.pixel_uv[1] <= bottom
        ]
        obstacle_hits = [ray for ray in inside if ray.surface == "obstacle"]
        candidates = obstacle_hits or inside
        if not candidates:
            return None, None, 0.0, "none"

        positions = [
            ray.world_position
            for ray in candidates
            if ray.world_position is not None
        ]
        distances = [
            max(
                0.05,
                float(ray.hit_distance_m or 0.0)
                * max(0.1, abs(ray.direction_camera[2])),
            )
            for ray in candidates
        ]
        if not positions or not distances:
            return None, None, 0.0, "none"
        world_position = [
            round(float(median(v[i] for v in positions)), 3) for i in range(3)
        ]
        depth_m = round(float(median(distances)), 3)
        sample_quality = min(1.0, 0.48 + len(candidates) * 0.12)
        semantic_quality = 1.0 if obstacle_hits else 0.72
        confidence = max(
            0.0,
            min(
                0.95,
                detection.confidence * sample_quality * semantic_quality,
            ),
        )
        return depth_m, world_position, round(confidence, 4), "unity_raycast"

    def _track_age(self, track_id: int | None, ts_ms: int) -> float:
        if track_id is None:
            return 0.0
        if not hasattr(self, "_track_created"):
            self._track_created: dict[int, int] = {}
        self._track_created.setdefault(track_id, ts_ms)
        return max(0.0, (ts_ms - self._track_created[track_id]) / 1000.0)

    def _deduplicate(self, detections: list[Detection]) -> list[Detection]:
        output: list[Detection] = []
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            if any(
                previous.label == detection.label
                and _iou(previous.bbox_xyxy, detection.bbox_xyxy) > 0.6
                for previous in output
            ):
                continue
            output.append(detection)
        return output

    def _compute_novelty(
        self,
        labels: set[str],
        centers: dict[int, tuple[float, float]],
    ) -> float:
        if not self._previous_labels and not self._previous_centers:
            return 0.0
        union = self._previous_labels | labels
        label_change = 1.0 - (
            len(self._previous_labels & labels) / len(union) if union else 1.0
        )
        all_ids = set(centers) | set(self._previous_centers)
        movement = 0.0
        if all_ids:
            movement = sum(
                min(
                    1.0,
                    math.hypot(
                        centers.get(track_id, (0.0, 0.0))[0]
                        - self._previous_centers.get(track_id, (0.0, 0.0))[0],
                        centers.get(track_id, (0.0, 0.0))[1]
                        - self._previous_centers.get(track_id, (0.0, 0.0))[1],
                    )
                    * 4.0,
                )
                for track_id in all_ids
            ) / len(all_ids)
        return max(label_change, movement)

    def _emit(self, payload: dict[str, object], dedupe_key: str) -> list[dict[str, object]]:
        if dedupe_key in self._event_history:
            return []
        self._event_history[dedupe_key] = None
        while len(self._event_history) > self._event_history_limit:
            self._event_history.popitem(last=False)
        return [payload]


def _iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0
