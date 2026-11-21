from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .contract import Detection, FrameEnvelope
from .tracking import IouTracker


class BackendUnavailable(RuntimeError):
    pass


class VisionBackend(Protocol):
    name: str

    @property
    def ready(self) -> bool: ...

    def process(self, frame: np.ndarray, envelope: FrameEnvelope) -> list[Detection]: ...

    def close(self) -> None: ...


@dataclass
class HeuristicBackend:
    """Offline foreground detector so the transport never depends on model files."""

    name: str = "heuristic_foreground"
    min_area_ratio: float = 0.0015
    max_area_ratio: float = 0.45
    _tracker: IouTracker | None = None

    @property
    def ready(self) -> bool:
        return True

    def process(self, frame: np.ndarray, envelope: FrameEnvelope) -> list[Detection]:
        if self._tracker is None:
            self._tracker = IouTracker()

        height, width = frame.shape[:2]
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        edges = cv2.Canny(blurred, 45, 115)

        # Remove long, straight geometry (horizon and rails) before finding
        # compact edge clusters. This is deliberately generic: unknown objects
        # are labelled "other", never assigned a guessed semantic class.
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=45,
            minLineLength=90,
            maxLineGap=18,
        )
        if lines is not None:
            for line in lines[:, 0]:
                start_x, start_y, end_x, end_y = (int(value) for value in line)
                length = float(np.hypot(end_x - start_x, end_y - start_y))
                if length > 120:
                    cv2.line(edges, (start_x, start_y), (end_x, end_y), 0, 5)
        mask = cv2.dilate(edges, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = float(width * height)
        proposals: list[Detection] = []
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            area_ratio = (box_width * box_height) / image_area
            if not self.min_area_ratio <= area_ratio <= self.max_area_ratio:
                continue
            if box_width < 18 or box_height < 18:
                continue
            aspect_ratio = box_width / max(1, box_height)
            if aspect_ratio > 4.0 or aspect_ratio < 0.18:
                continue
            if x <= 1 and box_width > width * 0.5:
                continue
            edge_count = int(cv2.countNonZero(edges[y : y + box_height, x : x + box_width]))
            density = edge_count / float(box_width * box_height)
            confidence = float(
                np.clip(0.34 + density * 1.4 + min(area_ratio, 0.08), 0.34, 0.72)
            )
            proposals.append(
                Detection(
                    label="other",
                    confidence=confidence,
                    bbox_xyxy=(float(x), float(y), float(x + box_width), float(y + box_height)),
                )
            )

        proposals.sort(key=lambda item: item.confidence, reverse=True)
        proposals = proposals[:12]
        return self._tracker.update(
            proposals, envelope.capture_ts_ms, envelope.camera.width, envelope.camera.height
        )

    def close(self) -> None:
        self._tracker = None


class YoloBackend:
    """YOLO detector with Ultralytics' persistent ByteTrack tracker."""

    def __init__(self, model_path: Path, *, confidence: float = 0.35, device: str | int = 0):
        if not model_path.exists():
            raise BackendUnavailable(f"model not found: {model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise BackendUnavailable("ultralytics is not installed") from error

        self._model = YOLO(str(model_path))
        self._confidence = confidence
        self._device = device
        self.name = f"yolo:{model_path.name}"

    @property
    def ready(self) -> bool:
        return True

    def process(self, frame: np.ndarray, envelope: FrameEnvelope) -> list[Detection]:
        results = self._model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self._confidence,
            iou=0.5,
            imgsz=640,
            verbose=False,
            device=self._device,
        )
        if not results:
            return []
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        identifiers = (
            boxes.id.detach().cpu().numpy().astype(int)
            if boxes.id is not None
            else np.full(len(boxes), -1, dtype=int)
        )
        names = result.names
        detections: list[Detection] = []
        for box, confidence, class_id, track_id in zip(
            xyxy, confidences, classes, identifiers, strict=False
        ):
            label = str(names.get(int(class_id), "other"))
            detections.append(
                Detection(
                    label=label,
                    confidence=float(confidence),
                    bbox_xyxy=tuple(float(value) for value in box[:4]),  # type: ignore[arg-type]
                    track_id=int(track_id) if track_id >= 0 else None,
                    label_source="fast",
                )
            )
        return detections

    def close(self) -> None:
        self._model = None
