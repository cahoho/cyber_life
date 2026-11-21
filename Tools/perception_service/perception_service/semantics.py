from __future__ import annotations

import base64
import json
import logging
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np

from .contract import FrameEnvelope, Snapshot


LOGGER = logging.getLogger("perception_service.semantics")


SCENE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "scene_summary": {"type": "string"},
        "place_type": {"type": "string"},
        "place_confidence": {"type": "number"},
        "notable_events": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
        "visible_landmarks": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "scene_summary",
        "place_type",
        "place_confidence",
        "notable_events",
        "uncertainties",
        "visible_landmarks",
    ],
}


ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string"},
        "grounded": {"type": "boolean"},
        "confidence": {"type": "number"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "grounded", "confidence", "evidence"],
}


SYSTEM_PROMPT = """你是具身角色的视觉理解模块。
只根据图片和给出的结构化观测回答。不要把没有像素证据的猜测写成事实。
几何、坐标、距离和可通行性以输入中的 navigation/objects 为准，不得修改。
地点类型是带置信度的假设；信息不足时明确说不确定。
使用简体中文回答。只输出符合给定 JSON schema 的 JSON。"""


class SemanticBackendError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SemanticSettings:
    api_key: str
    model: str
    base_url: str
    provider: str = "openai"
    timeout_s: float = 12.0
    refresh_s: float = 60.0

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def from_environment(cls) -> "SemanticSettings":
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        requested_provider = os.environ.get("SEMANTIC_PROVIDER", "auto").strip().lower()
        provider = (
            "deepseek"
            if requested_provider == "deepseek"
            or (requested_provider == "auto" and deepseek_key)
            else "openai"
        )
        if provider == "deepseek":
            api_key = deepseek_key
            model = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash").strip()
            base_url = os.environ.get(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
            ).rstrip("/")
        else:
            api_key = openai_key
            model = os.environ.get("OPENAI_VISION_MODEL", "gpt-4.1-mini").strip()
            base_url = os.environ.get(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            ).rstrip("/")
        return cls(
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_s=float(
                os.environ.get(
                    "SEMANTIC_TIMEOUT_S",
                    os.environ.get("OPENAI_VISION_TIMEOUT_S", "20"),
                )
            ),
            refresh_s=float(
                os.environ.get(
                    "SEMANTIC_REFRESH_S",
                    os.environ.get("OPENAI_VISION_REFRESH_S", "60"),
                )
            ),
        )


class SemanticAnalyzer:
    def __init__(self, settings: SemanticSettings) -> None:
        self.settings = settings

    @property
    def configured(self) -> bool:
        return self.settings.configured

    def analyze_scene(
        self,
        jpeg: bytes,
        snapshot: Snapshot,
        navigation: dict[str, Any] | None,
    ) -> dict[str, Any]:
        prompt = (
            "描述当前画面。重点判断角色目前可能位于哪种室内/室外空间"
            "（例如展馆、书房、办公室、走廊、客厅、厨房、商店或未知），"
            "列出可见地标和无法确认之处。"
            f"\n结构化观测：{_grounding_text(snapshot, navigation)}"
        )
        result = self._request(jpeg, prompt, SCENE_SCHEMA, "scene_understanding")
        place_type = str(result.get("place_type", "未知")).strip() or "未知"
        confidence = _clamp_float(result.get("place_confidence"), 0.0)
        return {
            "schema_version": "1.0",
            "source_frame_id": snapshot.frame.frame_id,
            "result_ts_ms": int(time.time() * 1000),
            "trigger": (
                "unknown_object"
                if snapshot.unknown_object_count
                else "scene_changed"
                if snapshot.major_change
                else "explicit_request"
            ),
            "scene_summary": str(result.get("scene_summary", "无法确定")),
            "place_type": place_type,
            "place_confidence": round(confidence, 4),
            "notable_events": _string_list(result.get("notable_events")),
            "uncertainties": _string_list(result.get("uncertainties")),
            "visible_landmarks": _string_list(result.get("visible_landmarks")),
            "backend": f"vlm:{self.settings.model}",
            "status": "ok",
        }

    def answer_question(
        self,
        question: str,
        jpeg: bytes,
        snapshot: Snapshot,
        navigation: dict[str, Any] | None,
        semantic: dict[str, Any] | None,
    ) -> dict[str, Any]:
        semantic_text = json.dumps(semantic, ensure_ascii=False) if semantic else "无"
        prompt = (
            f"用户问题：{question}\n"
            f"快速视觉与通行性：{_grounding_text(snapshot, navigation)}\n"
            f"最近一次场景理解：{semantic_text}\n"
            "如果问题询问画面里有没有某物，必须依据画面或给出现有检测；"
            "看不清就回答不确定。不要仅因为类别常见就猜测存在。"
        )
        result = self._request(jpeg, prompt, ANSWER_SCHEMA, "scene_answer")
        return {
            "question": question,
            "answer": str(result.get("answer", "无法确定")),
            "grounded": bool(result.get("grounded", False)),
            "confidence": round(_clamp_float(result.get("confidence"), 0.0), 4),
            "evidence": _string_list(result.get("evidence")),
            "source": "vlm",
            "source_frame_id": snapshot.frame.frame_id,
        }

    def _request(
        self,
        jpeg: bytes,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
    ) -> dict[str, Any]:
        if not self.configured:
            raise SemanticBackendError(
                "SLOW_BACKEND_UNAVAILABLE", "OPENAI_API_KEY is not configured"
            )
        image_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
        payload = {
            "model": self.settings.model,
            "instructions": SYSTEM_PROMPT,
            "store": False,
            "max_output_tokens": 700,
            "temperature": 0.1,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": image_url,
                            "detail": "high",
                        },
                    ],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        if self.settings.provider == "deepseek":
            payload["reasoning"] = {"effort": "none"}
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        endpoint = self.settings.base_url + "/responses"
        request = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.settings.timeout_s
                ) as response:
                    body = json.loads(response.read().decode("utf-8"))
                text = _response_text(body)
                parsed = json.loads(text)
                if not isinstance(parsed, dict):
                    raise ValueError("VLM output is not a JSON object")
                return parsed
            except urllib.error.HTTPError as error:
                detail = error.read(512).decode("utf-8", errors="replace")
                last_error = error
                if error.code not in {429, 500, 502, 503, 504} or attempt == 1:
                    raise SemanticBackendError(
                        "SLOW_BACKEND_ERROR", f"HTTP {error.code}: {detail}"
                    ) from error
                time.sleep(0.4)
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = error
                if attempt == 1:
                    raise SemanticBackendError(
                        "SLOW_TIMEOUT", f"VLM request failed: {error}"
                    ) from error
                time.sleep(0.4)
            except (KeyError, ValueError, json.JSONDecodeError) as error:
                raise SemanticBackendError(
                    "SLOW_BAD_OUTPUT", f"invalid VLM output: {error}"
                ) from error
        raise SemanticBackendError("SLOW_BACKEND_ERROR", str(last_error))


class SemanticWorker:
    """Low-frequency scene understanding; never blocks the fast frame path."""

    def __init__(
        self,
        analyzer: SemanticAnalyzer,
        on_result: Any,
        *,
        queue_size: int = 1,
    ) -> None:
        self._analyzer = analyzer
        self._on_result = on_result
        self._queue: queue.Queue[tuple[FrameEnvelope, bytes, Snapshot, dict[str, Any] | None] | None] = queue.Queue(
            maxsize=queue_size
        )
        self._thread = threading.Thread(
            target=self._run, name="semantic-worker", daemon=True
        )
        self._last_attempt_s = 0.0
        self._last_error: str | None = None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._put_latest(None)
        self._thread.join(timeout=5)

    def maybe_submit(
        self,
        image: np.ndarray,
        jpeg: bytes,
        frame: FrameEnvelope,
        snapshot: Snapshot,
        navigation: dict[str, Any] | None,
    ) -> None:
        if not self._analyzer.configured:
            return
        now = time.monotonic()
        high_priority = (
            frame.request_slow or snapshot.unknown_object_count > 0 or snapshot.major_change
        )
        minimum_interval = 15.0 if high_priority else self._analyzer.settings.refresh_s
        if self._last_attempt_s and now - self._last_attempt_s < minimum_interval:
            return
        self._last_attempt_s = now
        del image  # the compressed frame is enough and keeps queue memory bounded
        self._put_latest((frame, jpeg, snapshot, navigation))

    def _put_latest(
        self,
        item: tuple[FrameEnvelope, bytes, Snapshot, dict[str, Any] | None] | None,
    ) -> None:
        try:
            self._queue.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        self._queue.put(item, timeout=1.0)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            frame, jpeg, snapshot, navigation = item
            try:
                result = self._analyzer.analyze_scene(jpeg, snapshot, navigation)
                self._last_error = None
                self._on_result(frame.session_id, result)
            except Exception as error:
                self._last_error = self._safe_error(error)
                LOGGER.warning("scene understanding failed: %s", self._last_error)

    @staticmethod
    def _safe_error(error: Exception) -> str:
        if isinstance(error, SemanticBackendError):
            return f"{error.code}: {error}"
        return f"SLOW_BACKEND_ERROR: {error}"


def infer_place_hint(objects: list[dict[str, Any]]) -> tuple[str, float]:
    labels = {str(item.get("label", "")).lower() for item in objects}
    evidence: dict[str, int] = {
        "展馆或公共大厅": len(
            labels & {"vase", "potted plant", "bench", "person", "suitcase"}
        ),
        "书房或办公室": len(
            labels & {"book", "laptop", "keyboard", "mouse", "chair", "desk"}
        ),
        "客厅": len(labels & {"couch", "tv", "chair", "potted plant"}),
        "厨房": len(labels & {"bottle", "cup", "microwave", "oven", "bowl", "wine glass"}),
    }
    place, count = max(evidence.items(), key=lambda item: item[1])
    if count < 2:
        return "未知", 0.0
    return place, min(0.7, 0.25 + count * 0.12)


def _grounding_text(
    snapshot: Snapshot, navigation: dict[str, Any] | None
) -> str:
    objects = [
        {
            "track_id": item.get("track_id"),
            "label": item.get("label"),
            "confidence": item.get("confidence"),
            "center_uv": item.get("center_uv"),
        }
        for item in snapshot.objects
    ]
    nav_summary = navigation.get("summary") if navigation else None
    return json.dumps(
        {
            "frame_id": snapshot.frame.frame_id,
            "camera_position": list(snapshot.frame.camera.position),
            "objects": objects,
            "navigation": nav_summary,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _response_text(body: dict[str, Any]) -> str:
    direct = body.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    parts: list[str] = []
    for item in body.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict):
                text = content.get("text")
                if isinstance(text, str):
                    parts.append(text)
    if not parts:
        raise ValueError("Responses API output has no text content")
    return "".join(parts)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _clamp_float(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default
