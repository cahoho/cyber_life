from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .contract import CameraContract, NavigationObservation, RayObservation


@dataclass(frozen=True)
class _CellEvidence:
    hits: int = 0
    floor_hits: int = 0
    blocked_hits: int = 0
    min_obstacle_height_m: float | None = None
    evidence_uv: tuple[tuple[float, float], ...] = ()


def _quaternion_rotate(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    x, y, z, w = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _horizontal(vector: tuple[float, float, float]) -> tuple[float, float, float] | None:
    length = math.hypot(vector[0], vector[2])
    if length < 1e-6:
        return None
    return vector[0] / length, 0.0, vector[2] / length


def _infer_ground_y(
    camera: CameraContract, observation: NavigationObservation
) -> float:
    if observation.ground_y is not None:
        return observation.ground_y
    candidates = [
        ray.world_position[1]
        for ray in observation.rays
        if ray.hit
        and ray.world_position is not None
        and ray.world_normal is not None
        and ray.world_normal[1] >= math.cos(math.radians(observation.max_slope_deg))
    ]
    if not candidates:
        return camera.position[1] - 1.4
    candidates.sort()
    return candidates[len(candidates) // 2]


def _surface_from_geometry(
    ray: RayObservation,
    *,
    ground_y: float,
    max_slope_deg: float,
    step_height_m: float,
) -> str:
    if not ray.hit or ray.world_position is None or ray.world_normal is None:
        return "unknown"
    normal = ray.world_normal
    horizontal = math.hypot(normal[0], normal[2])
    slope_deg = math.degrees(math.atan2(horizontal, max(1e-6, abs(normal[1]))))
    height_m = ray.world_position[1] - ground_y
    if normal[1] < 0:
        slope_deg = 180.0 - slope_deg
    if normal[1] >= math.cos(math.radians(max_slope_deg)):
        if height_m <= step_height_m:
            return "floor"
        return "obstacle"
    if height_m > 2.0:
        return "overhead"
    return "obstacle"


def build_navigation(
    camera: CameraContract,
    observation: NavigationObservation | None,
    *,
    forward_m: float = 8.0,
    half_width_m: float = 3.0,
    cell_size_m: float = 0.5,
) -> dict[str, Any] | None:
    """Turn camera ray hits into a body-local traversability grid.

    The grid coordinates are ``right`` and ``forward`` on the camera's ground
    projection. Geometry is produced locally; a VLM never writes these cells.
    """

    if observation is None:
        return None

    forward_axis = _horizontal(_quaternion_rotate(camera.rotation_xyzw, (0.0, 0.0, 1.0)))
    right_axis = _horizontal(_quaternion_rotate(camera.rotation_xyzw, (1.0, 0.0, 0.0)))
    if forward_axis is None or right_axis is None:
        return None

    ground_y = _infer_ground_y(camera, observation)
    rows = max(1, int(math.ceil(forward_m / cell_size_m)))
    columns = max(1, int(math.ceil((half_width_m * 2.0) / cell_size_m)))
    evidence: dict[tuple[int, int], _CellEvidence] = {}

    for ray in observation.rays:
        if not ray.hit or ray.world_position is None:
            continue
        delta = (
            ray.world_position[0] - camera.position[0],
            0.0,
            ray.world_position[2] - camera.position[2],
        )
        local_forward = delta[0] * forward_axis[0] + delta[2] * forward_axis[2]
        local_right = delta[0] * right_axis[0] + delta[2] * right_axis[2]
        if not -half_width_m <= local_right <= half_width_m:
            continue
        if not 0 < local_forward <= forward_m:
            continue

        row = min(rows - 1, int(local_forward / cell_size_m))
        column = min(columns - 1, int((local_right + half_width_m) / cell_size_m))
        key = (row, column)
        previous = evidence.get(key, _CellEvidence())
        surface = _surface_from_geometry(
            ray,
            ground_y=ground_y,
            max_slope_deg=observation.max_slope_deg,
            step_height_m=observation.step_height_m,
        )
        floor_hits = previous.floor_hits + (1 if surface == "floor" else 0)
        blocked_hits = previous.blocked_hits + (1 if surface == "obstacle" else 0)
        obstacle_height = ray.world_position[1] - ground_y if surface == "obstacle" else None
        min_height = previous.min_obstacle_height_m
        if obstacle_height is not None:
            min_height = (
                obstacle_height
                if min_height is None
                else min(min_height, obstacle_height)
            )
        evidence[key] = _CellEvidence(
            hits=previous.hits + 1,
            floor_hits=floor_hits,
            blocked_hits=blocked_hits,
            min_obstacle_height_m=min_height,
            evidence_uv=previous.evidence_uv + (ray.pixel_uv,),
        )

    cells: list[dict[str, Any]] = []
    for row in range(rows):
        for column in range(columns):
            item = evidence.get((row, column))
            center_right = -half_width_m + (column + 0.5) * cell_size_m
            center_forward = (row + 0.5) * cell_size_m
            if item is None:
                status = "unknown"
                confidence = 0.0
                obstacle_height = None
            elif item.blocked_hits:
                status = "blocked"
                confidence = min(0.98, 0.55 + 0.1 * item.blocked_hits)
                obstacle_height = item.min_obstacle_height_m
            elif item.floor_hits:
                status = "traversable"
                confidence = min(0.98, 0.5 + 0.1 * item.floor_hits)
                obstacle_height = None
            else:
                status = "unknown"
                confidence = 0.15
                obstacle_height = None
            world_center = [
                camera.position[0]
                + right_axis[0] * center_right
                + forward_axis[0] * center_forward,
                ground_y,
                camera.position[2]
                + right_axis[2] * center_right
                + forward_axis[2] * center_forward,
            ]
            cells.append(
                {
                    "row": row,
                    "column": column,
                    "center_right_m": round(center_right, 3),
                    "center_forward_m": round(center_forward, 3),
                    "world_center": [round(value, 3) for value in world_center],
                    "status": status,
                    "confidence": round(confidence, 3),
                    "obstacle_height_m": (
                        None if obstacle_height is None else round(obstacle_height, 3)
                    ),
                    "evidence_uv": [
                        [round(uv[0], 4), round(uv[1], 4)]
                        for uv in (item.evidence_uv[:4] if item is not None else ())
                    ],
                    "evidence_u": [
                        round(uv[0], 4)
                        for uv in (item.evidence_uv[:4] if item is not None else ())
                    ],
                    "evidence_v": [
                        round(uv[1], 4)
                        for uv in (item.evidence_uv[:4] if item is not None else ())
                    ],
                }
            )

    involved = [cell for cell in cells if cell["status"] != "unknown"]
    traversable = sum(cell["status"] == "traversable" for cell in involved)
    directions = _directions(cells, rows, columns, cell_size_m, half_width_m, forward_m)
    best = max(directions, key=lambda item: (item["clearance_m"], -abs(item["heading_deg"])))
    return {
        "schema_version": "1.0",
        "source": observation.source,
        "frame_id": None,
        "coordinate_frame": "camera_ground",
        "ground_y": round(ground_y, 3),
        "grid": {
            "cell_size_m": cell_size_m,
            "forward_m": forward_m,
            "half_width_m": half_width_m,
            "rows": rows,
            "columns": columns,
        },
        "cells": cells,
        "directions": directions,
        "summary": {
            "traversable_cell_count": sum(
                cell["status"] == "traversable" for cell in cells
            ),
            "blocked_cell_count": sum(cell["status"] == "blocked" for cell in cells),
            "unknown_cell_count": sum(cell["status"] == "unknown" for cell in cells),
            "traversable_ratio": round(
                traversable / max(1, len(involved)),
                3,
            ),
            "forward_clearance_m": directions[len(directions) // 2]["clearance_m"],
            "recommended_heading_deg": best["heading_deg"],
            "recommended_clearance_m": best["clearance_m"],
        },
    }


def _directions(
    cells: list[dict[str, Any]],
    rows: int,
    columns: int,
    cell_size_m: float,
    half_width_m: float,
    forward_m: float,
) -> list[dict[str, Any]]:
    lookup = {(cell["row"], cell["column"]): cell for cell in cells}
    result: list[dict[str, Any]] = []
    for heading in range(-60, 61, 15):
        radians = math.radians(heading)
        blocked_forward = forward_m
        for row in range(rows):
            distance = (row + 0.5) * cell_size_m
            right = math.tan(radians) * distance
            column = int((right + half_width_m) / cell_size_m)
            if not 0 <= column < columns:
                blocked_forward = min(blocked_forward, distance)
                break
            if lookup[(row, column)]["status"] == "blocked":
                blocked_forward = min(blocked_forward, max(0.0, distance - cell_size_m * 0.5))
                break
        result.append(
            {
                "heading_deg": heading,
                "clearance_m": round(blocked_forward, 2),
                "status": "clear" if blocked_forward >= forward_m else "limited",
            }
        )
    return result
