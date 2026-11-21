from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from .contract import FrameEnvelope, Snapshot
from .semantics import SemanticAnalyzer, SemanticBackendError


OBJECT_ALIASES: dict[str, tuple[str, ...]] = {
    "人": ("person",),
    "行人": ("person",),
    "椅子": ("chair",),
    "凳子": ("chair",),
    "桌子": ("dining table", "table", "desk"),
    "餐桌": ("dining table", "table"),
    "书": ("book",),
    "书桌": ("desk", "dining table"),
    "电脑": ("laptop", "computer", "tv"),
    "笔记本": ("laptop",),
    "杯子": ("cup",),
    "瓶子": ("bottle",),
    "花瓶": ("vase",),
    "植物": ("potted plant", "plant"),
    "盆栽": ("potted plant", "plant"),
    "沙发": ("couch", "sofa"),
    "电视": ("tv", "television"),
    "门": ("door",),
    "汽车": ("car",),
    "包": ("backpack", "handbag", "suitcase"),
    "行李箱": ("suitcase",),
}


@dataclass
class FrameRecord:
    frame: FrameEnvelope
    jpeg: bytes
    snapshot: Snapshot
    navigation: dict[str, Any] | None
    received_monotonic: float


class SceneMemory:
    def __init__(self, analyzer: SemanticAnalyzer) -> None:
        self._analyzer = analyzer
        self._lock = threading.Lock()
        self._latest: FrameRecord | None = None
        self._semantic: dict[str, Any] | None = None
        self._semantic_session: str | None = None
        self._last_query_error: str | None = None

    def update(
        self,
        frame: FrameEnvelope,
        jpeg: bytes,
        snapshot: Snapshot,
        navigation: dict[str, Any] | None,
    ) -> None:
        with self._lock:
            self._latest = FrameRecord(
                frame=frame,
                jpeg=jpeg,
                snapshot=snapshot,
                navigation=navigation,
                received_monotonic=time.monotonic(),
            )
            if self._semantic_session != frame.session_id:
                self._semantic = None
                self._semantic_session = frame.session_id

    def set_semantic(self, session_id: str, semantic: dict[str, Any]) -> None:
        with self._lock:
            if self._latest is not None and session_id == self._latest.frame.session_id:
                self._semantic_session = session_id
                self._semantic = semantic

    def snapshot(self) -> FrameRecord | None:
        with self._lock:
            return self._latest

    def semantic(self) -> dict[str, Any] | None:
        with self._lock:
            return self._semantic

    def state(self) -> dict[str, Any]:
        with self._lock:
            record = self._latest
            semantic = self._semantic
        if record is None:
            return {
                "status": "waiting_for_frame",
                "vlm_configured": self._analyzer.configured,
            }
        return {
            "status": "ready",
            "vlm_configured": self._analyzer.configured,
            "vlm_provider": self._analyzer.settings.provider,
            "vlm_model": self._analyzer.settings.model,
            "frame_id": record.frame.frame_id,
            "capture_ts_ms": record.frame.capture_ts_ms,
            "age_ms": round((time.monotonic() - record.received_monotonic) * 1000),
            "camera": record.frame.camera.to_result_dict(),
            "objects": record.snapshot.objects,
            "place_hint": record.snapshot.place_hint,
            "place_confidence": record.snapshot.place_confidence,
            "navigation": (
                record.navigation.get("summary")
                if record.navigation is not None
                else None
            ),
            "semantic": semantic,
            "last_query_error": self._last_query_error,
        }

    def ask(self, question: str) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        with self._lock:
            record = self._latest
        if record is None:
            return {
                "question": question,
                "answer": "还没有收到视觉帧，暂时无法回答。",
                "grounded": False,
                "confidence": 0.0,
                "evidence": [],
                "source": "state",
            }

        if self._analyzer.configured:
            try:
                answer = self._analyzer.answer_question(
                    question,
                    record.jpeg,
                    record.snapshot,
                    record.navigation,
                    self.semantic(),
                )
                self._last_query_error = None
                return answer
            except SemanticBackendError as error:
                self._last_query_error = f"{error.code}: {error}"

        return self._answer_locally(question, record)

    def _answer_locally(self, question: str, record: FrameRecord) -> dict[str, Any]:
        normalized = re.sub(r"\s+", "", question.lower())
        objects = record.snapshot.objects
        nav_summary = (
            record.navigation.get("summary") if record.navigation is not None else None
        )
        location_terms = ("我在哪", "当前位置", "我的位置", "whatplace", "whereami", "location")
        place_terms = ("什么地方", "什么场景", "哪里", "场所", "place", "scene")
        obstacle_terms = (
            "障碍",
            "挡住",
            "能走",
            "可通行",
            "通行",
            "怎么走",
            "通路",
            "obstacle",
            "blocked",
            "clear",
        )
        list_terms = ("有什么", "看到什么", "看见什么", "whatdoyousee", "objects")

        if any(term in normalized for term in location_terms):
            p = record.frame.camera.position
            return self._local_answer(
                question,
                f"当前位置约为 x={p[0]:.2f}, y={p[1]:.2f}, z={p[2]:.2f} 米；"
                f"当前地点判断为“{record.snapshot.place_hint or '未知'}”。",
                ["camera.position", "scene.place_hint"],
                min(0.9, record.snapshot.place_confidence + 0.1),
            )
        if any(term in normalized for term in place_terms):
            return self._local_answer(
                question,
                f"当前画面更像“{record.snapshot.place_hint or '未知'}”，"
                f"置信度约 {record.snapshot.place_confidence:.2f}。",
                ["scene.place_hint"],
                record.snapshot.place_confidence,
            )
        if any(term in normalized for term in obstacle_terms):
            if nav_summary is None:
                answer = "本帧没有几何射线数据，无法可靠判断通行性。"
                confidence = 0.0
            else:
                answer = (
                    f"前方净空约 {nav_summary['forward_clearance_m']:.1f} 米，"
                    f"检测到 {nav_summary['blocked_cell_count']} 个障碍网格；"
                    f"建议朝向约 {nav_summary['recommended_heading_deg']} 度，"
                    f"净空约 {nav_summary['recommended_clearance_m']:.1f} 米。"
                )
                confidence = 0.75
            return self._local_answer(
                question, answer, ["navigation.summary"], confidence
            )
        if any(term in normalized for term in list_terms):
            labels = sorted({str(item.get("label")) for item in objects})
            answer = (
                "当前快检测到：" + "、".join(labels) + "。"
                if labels
                else "当前快检测没有识别到明确物体。"
            )
            return self._local_answer(question, answer, ["objects[].label"], 0.65)

        aliases = _aliases_in_question(normalized)
        if aliases:
            matches = [
                item
                for item in objects
                if str(item.get("label", "")).lower() in aliases
            ]
            if matches:
                labels = sorted({str(item["label"]) for item in matches})
                answer = f"检测到 {len(matches)} 个相关目标：{'、'.join(labels)}。"
                confidence = max(float(item.get("confidence", 0.0)) for item in matches)
                return self._local_answer(
                    question, answer, ["objects[].label"], confidence
                )
            return self._local_answer(
                question,
                "当前帧没有检测到该类目标；这不等于视野外一定不存在。",
                ["objects[].label"],
                0.45,
            )

        return self._local_answer(
            question,
            "未配置视觉语言 API 时只能回答物体是否存在、通行性和当前位置等结构化问题。"
            "复杂语义问题需要 OPENAI_API_KEY。",
            ["local_fast_state"],
            0.3,
        )

    @staticmethod
    def _local_answer(
        question: str, answer: str, evidence: list[str], confidence: float
    ) -> dict[str, Any]:
        return {
            "question": question,
            "answer": answer,
            "grounded": True,
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
            "evidence": evidence,
            "source": "fast_state",
        }


def _aliases_in_question(normalized: str) -> set[str]:
    result: set[str] = set()
    for chinese, labels in OBJECT_ALIASES.items():
        if chinese in normalized:
            result.update(labels)
    for aliases in OBJECT_ALIASES.values():
        for label in aliases:
            if re.search(rf"(?<![a-z]){re.escape(label)}(?![a-z])", normalized):
                result.add(label)
    return result
