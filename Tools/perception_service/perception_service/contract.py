from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SCHEMA_VERSION = "1.0"


class ContractError(ValueError):
    """Raised when an incoming frame violates the wire contract."""


def _number_list(
    value: Any,
    name: str,
    length: int,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ContractError(f"{name} must be an array of {length} numbers")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ContractError(f"{name} contains a non-numeric value")
        number = float(item)
        if minimum is not None and number < minimum:
            raise ContractError(f"{name} contains a value below {minimum}")
        if maximum is not None and number > maximum:
            raise ContractError(f"{name} contains a value above {maximum}")
        result.append(number)
    return tuple(result)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an integer")
    return value


@dataclass(frozen=True)
class CameraContract:
    width: int
    height: int
    fov_y_deg: float
    near: float
    far: float
    position: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    unity_to_cv: bool

    @classmethod
    def from_dict(cls, data: Any) -> "CameraContract":
        if not isinstance(data, dict):
            raise ContractError("camera must be an object")
        width = _integer(data.get("width"), "camera.width")
        height = _integer(data.get("height"), "camera.height")
        if width <= 0 or height <= 0:
            raise ContractError("camera dimensions must be positive")
        fov = float(data.get("fov_y_deg", 0))
        if not 0 < fov < 180:
            raise ContractError("camera.fov_y_deg must be between 0 and 180")
        position = _number_list(data.get("position"), "camera.position", 3)
        rotation = _number_list(
            data.get("rotation_xyzw"), "camera.rotation_xyzw", 4
        )
        quaternion_norm = sum(component * component for component in rotation)
        if not 0.5 < quaternion_norm < 1.5:
            raise ContractError("camera rotation must be a normalized quaternion")
        return cls(
            width=width,
            height=height,
            fov_y_deg=fov,
            near=float(data.get("near", 0.1)),
            far=float(data.get("far", 1000.0)),
            position=position,  # type: ignore[arg-type]
            rotation_xyzw=rotation,  # type: ignore[arg-type]
            unity_to_cv=bool(data.get("unity_to_cv", True)),
        )

    def to_result_dict(self) -> dict[str, list[float]]:
        return {
            "position": list(self.position),
            "rotation_xyzw": list(self.rotation_xyzw),
        }


@dataclass(frozen=True)
class RayObservation:
    """One camera-space ray sample supplied by Unity physics."""

    pixel_uv: tuple[float, float]
    direction_camera: tuple[float, float, float]
    hit: bool
    hit_distance_m: float | None
    world_position: tuple[float, float, float] | None
    world_normal: tuple[float, float, float] | None
    surface: str

    @classmethod
    def from_dict(cls, data: Any) -> "RayObservation":
        if not isinstance(data, dict):
            raise ContractError("navigation.rays[] must be an object")
        pixel_uv = _number_list(data.get("pixel_uv"), "navigation.rays[].pixel_uv", 2)
        direction = _number_list(
            data.get("direction_camera"), "navigation.rays[].direction_camera", 3
        )
        direction_norm = sum(value * value for value in direction)
        if not 0.5 < direction_norm < 1.5:
            raise ContractError("navigation ray direction must be near unit length")
        hit = bool(data.get("hit", False))
        world_position = None
        world_normal = None
        hit_distance = None
        if hit:
            world_position = _number_list(
                data.get("world_position"),
                "navigation.rays[].world_position",
                3,
            )
            world_normal = _number_list(
                data.get("world_normal"),
                "navigation.rays[].world_normal",
                3,
            )
            normal_norm = sum(value * value for value in world_normal)
            if not 0.5 < normal_norm < 1.5:
                raise ContractError("navigation ray normal must be near unit length")
            hit_distance = float(data.get("hit_distance_m", 0.0))
            if hit_distance <= 0:
                raise ContractError("navigation hit distance must be positive")
        surface = str(data.get("surface", "unknown"))
        if surface not in {"floor", "obstacle", "overhead", "unknown"}:
            raise ContractError(f"unsupported navigation surface {surface!r}")
        return cls(
            pixel_uv=(pixel_uv[0], pixel_uv[1]),
            direction_camera=(direction[0], direction[1], direction[2]),
            hit=hit,
            hit_distance_m=hit_distance,
            world_position=world_position,
            world_normal=world_normal,
            surface=surface,
        )


@dataclass(frozen=True)
class NavigationObservation:
    rays: tuple[RayObservation, ...]
    source: str = "unity_raycast"
    ground_y: float | None = None
    max_slope_deg: float = 45.0
    step_height_m: float = 0.20

    @classmethod
    def from_dict(cls, data: Any) -> "NavigationObservation":
        if not isinstance(data, dict):
            raise ContractError("navigation must be an object")
        raw_rays = data.get("rays")
        if not isinstance(raw_rays, list) or not raw_rays:
            raise ContractError("navigation.rays must be a non-empty array")
        if len(raw_rays) > 1024:
            raise ContractError("navigation.rays exceeds 1024 samples")
        ground = data.get("ground_y")
        ground_y = None if ground is None else float(ground)
        max_slope = float(data.get("max_slope_deg", 45.0))
        if not 0 < max_slope < 89:
            raise ContractError("navigation.max_slope_deg must be between 0 and 89")
        step_height = float(data.get("step_height_m", 0.20))
        if not 0 <= step_height <= 1.0:
            raise ContractError("navigation.step_height_m must be in [0, 1]")
        return cls(
            rays=tuple(RayObservation.from_dict(item) for item in raw_rays),
            source=str(data.get("source", "unity_raycast")),
            ground_y=ground_y,
            max_slope_deg=max_slope,
            step_height_m=step_height,
        )


@dataclass(frozen=True)
class FrameEnvelope:
    session_id: str
    frame_id: int
    capture_ts_ms: int
    camera: CameraContract
    request_fast: bool
    request_slow: bool
    navigation: NavigationObservation | None = None
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: Any) -> "FrameEnvelope":
        if not isinstance(data, dict):
            raise ContractError("metadata must be a JSON object")
        schema_version = str(data.get("schema_version", ""))
        if schema_version != SCHEMA_VERSION:
            raise ContractError(
                f"unsupported schema_version {schema_version!r}; expected {SCHEMA_VERSION!r}"
            )
        session_id = data.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ContractError("session_id must be a non-empty string")
        request = data.get("request", {})
        if not isinstance(request, dict):
            raise ContractError("request must be an object")
        raw_navigation = data.get("navigation")
        return cls(
            session_id=session_id,
            frame_id=_integer(data.get("frame_id"), "frame_id"),
            capture_ts_ms=_integer(data.get("capture_ts_ms"), "capture_ts_ms"),
            camera=CameraContract.from_dict(data.get("camera")),
            request_fast=bool(request.get("fast", True)),
            request_slow=bool(request.get("slow", False)),
            navigation=(
                NavigationObservation.from_dict(raw_navigation)
                if raw_navigation is not None
                else None
            ),
            schema_version=schema_version,
        )


@dataclass
class TrackState:
    track_id: int
    label: str
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    created_ts_ms: int
    last_seen_ts_ms: int
    last_center_uv: tuple[float, float]
    velocity_uv_per_s: tuple[float, float] = (0.0, 0.0)
    hits: int = 1

    @property
    def age_s(self) -> float:
        return max(0.0, (self.last_seen_ts_ms - self.created_ts_ms) / 1000.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class Detection:
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    track_id: int | None = None
    label_source: str = "fast"


@dataclass
class Snapshot:
    frame: FrameEnvelope
    result_ts_ms: int
    infer_ms: int
    objects: list[dict[str, Any]] = field(default_factory=list)
    novelty: float = 0.0
    major_change: bool = False
    unknown_object_count: int = 0
    vision_events: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    backend: str = "unknown"
    navigation: dict[str, Any] | None = None
    place_hint: str | None = None
    place_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.frame.session_id,
            "source_frame_id": self.frame.frame_id,
            "capture_ts_ms": self.frame.capture_ts_ms,
            "result_ts_ms": self.result_ts_ms,
            "infer_ms": self.infer_ms,
            "camera_pose_at_capture": self.frame.camera.to_result_dict(),
            "objects": self.objects,
            "scene": {
                "novelty": round(self.novelty, 4),
                "major_change": self.major_change,
                "unknown_object_count": self.unknown_object_count,
                "place_hint": self.place_hint,
                "place_confidence": round(self.place_confidence, 4),
            },
            "vision_events": self.vision_events,
            "warnings": self.warnings,
            "backend": self.backend,
            "navigation": self.navigation,
        }


def event(event_type: str, ts_ms: int, track_id: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": event_type, "ts_ms": ts_ms, "payload": {}}
    if track_id is not None:
        payload["track_id"] = track_id
    return payload
