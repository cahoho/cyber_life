from __future__ import annotations

from collections.abc import Iterable

from .contract import Detection, TrackState


def iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0:
        return 0.0
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


class IouTracker:
    """Small deterministic tracker used by the heuristic backend."""

    def __init__(self, iou_gate: float = 0.2, lost_after_ms: int = 2000) -> None:
        self._iou_gate = iou_gate
        self._lost_after_ms = lost_after_ms
        self._tracks: dict[int, TrackState] = {}
        self._next_id = 1

    @property
    def tracks(self) -> tuple[TrackState, ...]:
        return tuple(self._tracks.values())

    def update(
        self, detections: Iterable[Detection], ts_ms: int, width: int, height: int
    ) -> list[Detection]:
        pending = list(detections)
        matched_ids: set[int] = set()
        output: list[Detection] = []

        candidates: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(pending):
            for track in self._tracks.values():
                if track.label != detection.label:
                    continue
                score = iou(track.bbox_xyxy, detection.bbox_xyxy)
                if score >= self._iou_gate:
                    candidates.append((score, detection_index, track.track_id))
        candidates.sort(reverse=True)

        assigned_detections: set[int] = set()
        assignments: dict[int, int] = {}
        for _, detection_index, track_id in candidates:
            if detection_index in assigned_detections or track_id in matched_ids:
                continue
            assigned_detections.add(detection_index)
            matched_ids.add(track_id)
            assignments[detection_index] = track_id

        for index, detection in enumerate(pending):
            track_id = assignments.get(index)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                center_u = ((detection.bbox_xyxy[0] + detection.bbox_xyxy[2]) / 2) / width
                center_v = ((detection.bbox_xyxy[1] + detection.bbox_xyxy[3]) / 2) / height
                track = TrackState(
                    track_id=track_id,
                    label=detection.label,
                    bbox_xyxy=detection.bbox_xyxy,
                    confidence=detection.confidence,
                    created_ts_ms=ts_ms,
                    last_seen_ts_ms=ts_ms,
                    last_center_uv=(center_u, center_v),
                )
                self._tracks[track_id] = track
            else:
                previous = self._tracks[track_id]
                center_u = ((detection.bbox_xyxy[0] + detection.bbox_xyxy[2]) / 2) / width
                center_v = ((detection.bbox_xyxy[1] + detection.bbox_xyxy[3]) / 2) / height
                dt = max(0.001, (ts_ms - previous.last_seen_ts_ms) / 1000.0)
                previous.velocity_uv_per_s = (
                    (center_u - previous.last_center_uv[0]) / dt,
                    (center_v - previous.last_center_uv[1]) / dt,
                )
                previous.last_center_uv = (center_u, center_v)
                previous.bbox_xyxy = detection.bbox_xyxy
                previous.confidence = detection.confidence
                previous.last_seen_ts_ms = ts_ms
                previous.hits += 1
            output.append(
                Detection(
                    label=detection.label,
                    confidence=detection.confidence,
                    bbox_xyxy=detection.bbox_xyxy,
                    track_id=track_id,
                    label_source=detection.label_source,
                )
            )

        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if ts_ms - track.last_seen_ts_ms <= self._lost_after_ms
        }
        return output
