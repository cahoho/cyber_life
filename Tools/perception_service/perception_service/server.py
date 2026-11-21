from __future__ import annotations

import argparse
import json
import logging
import queue
import threading
from dataclasses import dataclass, field
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

from .backends import BackendUnavailable, HeuristicBackend, VisionBackend, YoloBackend
from .config import load_local_environment
from .contract import ContractError, FrameEnvelope, Snapshot
from .engine import VisionEngine
from .semantics import SemanticAnalyzer, SemanticSettings, SemanticWorker
from .state import SceneMemory


LOGGER = logging.getLogger("perception_service")
MAX_REQUEST_BYTES = 1_500_000


@dataclass
class Job:
    envelope: FrameEnvelope
    image: np.ndarray
    complete: threading.Event = field(default_factory=threading.Event)
    snapshot: Snapshot | None = None
    error: str | None = None


class FrameWorker:
    def __init__(self, engine: VisionEngine, queue_size: int = 2) -> None:
        self._engine = engine
        self._queue: queue.Queue[Job | None] = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._run, name="vision-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._put_latest(None)
        self._thread.join(timeout=5)
        self._engine.backend.close()

    def submit(self, envelope: FrameEnvelope, image: np.ndarray) -> Snapshot:
        job = Job(envelope=envelope, image=image)
        self._put_latest(job)
        if not job.complete.wait(timeout=8.0):
            raise TimeoutError("PERCEPTION_TIMEOUT: worker did not finish within 8s")
        if job.error is not None:
            raise RuntimeError(job.error)
        if job.snapshot is None:
            raise RuntimeError("PERCEPTION_INTERNAL_ERROR: missing snapshot")
        return job.snapshot

    def _put_latest(self, job: Job | None) -> None:
        try:
            self._queue.put_nowait(job)
            return
        except queue.Full:
            pass
        try:
            dropped = self._queue.get_nowait()
            if dropped is not None:
                dropped.error = "PERCEPTION_BACKPRESSURE: superseded by newer frame"
                dropped.complete.set()
        except queue.Empty:
            pass
        self._queue.put(job, timeout=1.0)

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                job.snapshot = self._engine.process(job.image, job.envelope)
            except Exception as error:  # keep worker alive after one bad frame
                LOGGER.exception("frame processing failed")
                job.error = f"PERCEPTION_INFERENCE_ERROR: {error}"
            finally:
                job.complete.set()


class PerceptionHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        worker: FrameWorker,
        backend_name: str,
        memory: SceneMemory,
        semantic_worker: SemanticWorker,
        semantic_analyzer: SemanticAnalyzer,
    ):
        super().__init__(address, PerceptionRequestHandler)
        self.worker = worker
        self.backend_name = backend_name
        self.memory = memory
        self.semantic_worker = semantic_worker
        self.semantic_analyzer = semantic_analyzer


class PerceptionRequestHandler(BaseHTTPRequestHandler):
    server: PerceptionHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        LOGGER.info("%s - %s", self.address_string(), format % args)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/health", "/v1/health"}:
            self._json(
                HTTPStatus.OK,
                {
                    "status": "ready",
                    "schema_version": "1.0",
                    "fast_model": self.server.backend_name,
                    "slow_backend": (
                        f"vlm:{self.server.semantic_analyzer.settings.provider}:"
                        f"{self.server.semantic_analyzer.settings.model}"
                        if self.server.semantic_analyzer.configured
                        else "none"
                    ),
                    "vlm_provider": self.server.semantic_analyzer.settings.provider,
                    "vlm_model": self.server.semantic_analyzer.settings.model,
                    "vlm_configured": self.server.semantic_analyzer.configured,
                },
            )
            return
        if path in {"/v1/state", "/state"}:
            self._json(HTTPStatus.OK, self.server.memory.state())
            return
        if path == "/":
            self._json(
                HTTPStatus.OK,
                {
                    "service": "CyberLife Perception",
                    "frame_endpoint": "POST /v1/frames",
                    "query_endpoint": "POST /v1/query",
                    "state_endpoint": "GET /v1/state",
                    "health": "/health",
                },
            )
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "NOT_FOUND"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/v1/query":
            self._handle_query()
            return
        if path != "/v1/frames":
            self._json(HTTPStatus.NOT_FOUND, {"error": "NOT_FOUND"})
            return
        try:
            metadata, image = self._read_multipart()
            envelope = FrameEnvelope.from_dict(metadata)
            decoded = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ContractError("image is not a decodable JPEG")
            height, width = decoded.shape[:2]
            if (width, height) != (envelope.camera.width, envelope.camera.height):
                decoded = cv2.resize(
                    decoded,
                    (envelope.camera.width, envelope.camera.height),
                    interpolation=cv2.INTER_AREA,
                )
            snapshot = self.server.worker.submit(envelope, decoded)
            self.server.memory.update(
                envelope,
                image,
                snapshot,
                snapshot.navigation,
            )
            self.server.semantic_worker.maybe_submit(
                decoded,
                image,
                envelope,
                snapshot,
                snapshot.navigation,
            )
            self._json(HTTPStatus.OK, snapshot.to_dict())
        except ContractError as error:
            LOGGER.warning("query rejected: %s", error)
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except TimeoutError as error:
            self._json(HTTPStatus.GATEWAY_TIMEOUT, {"error": str(error)})
        except Exception as error:
            LOGGER.exception("request failed")
            self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"PERCEPTION_INTERNAL_ERROR: {error}"},
            )

    def _handle_query(self) -> None:
        try:
            payload = self._read_json_body()
            question = payload.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ContractError("question must be a non-empty string")
            if len(question) > 2000:
                raise ContractError("question exceeds 2000 characters")
            answer = self.server.memory.ask(question)
            self._json(HTTPStatus.OK, answer)
        except ContractError as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception as error:
            LOGGER.exception("query failed")
            self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"SLOW_BACKEND_ERROR: {error}"},
            )

    def _read_json_body(self) -> dict[str, Any]:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("application/json"):
            raise ContractError("Content-Type must be application/json")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ContractError("invalid Content-Length") from error
        if not 0 < length <= 64_000:
            raise ContractError("JSON body must be between 1 and 64000 bytes")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContractError("body is not valid UTF-8 JSON") from error
        if not isinstance(payload, dict):
            raise ContractError("body must be a JSON object")
        return payload

    def _read_multipart(self) -> tuple[dict[str, Any], bytes]:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            raise ContractError("Content-Type must be multipart/form-data")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ContractError("invalid Content-Length") from error
        if not 0 < length <= MAX_REQUEST_BYTES:
            raise ContractError(
                f"request body must be between 1 and {MAX_REQUEST_BYTES} bytes"
            )
        body = self.rfile.read(length)
        synthetic = (
            b"Content-Type: "
            + content_type.encode("ascii", errors="replace")
            + b"\r\nMIME-Version: 1.0\r\n\r\n"
            + body
        )
        message = BytesParser(policy=policy.default).parsebytes(synthetic)
        if not message.is_multipart():
            raise ContractError("malformed multipart body")

        parts: dict[str, bytes] = {}
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if name:
                payload = part.get_payload(decode=True)
                if payload is not None:
                    parts[str(name)] = payload
        if "metadata" not in parts or "image" not in parts:
            raise ContractError("multipart body requires metadata and image parts")
        if len(parts["image"]) > 400_000:
            raise ContractError("TRANSPORT_OVERSIZE: JPEG exceeds 400 KB")
        if len(parts["metadata"]) > 262_144:
            raise ContractError("TRANSPORT_OVERSIZE: metadata exceeds 256 KB")
        try:
            metadata = json.loads(parts["metadata"].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContractError("metadata is not valid UTF-8 JSON") from error
        return metadata, parts["image"]

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


def build_backend(
    requested: str, model_path: Path, confidence: float, device: str
) -> VisionBackend:
    if requested == "heuristic":
        return HeuristicBackend()
    try:
        return YoloBackend(model_path, confidence=confidence, device=device)
    except BackendUnavailable as error:
        if requested == "yolo":
            raise
        LOGGER.warning("%s; falling back to heuristic backend", error)
        return HeuristicBackend()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CyberLife visual understanding service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8710)
    parser.add_argument(
        "--backend",
        choices=("auto", "yolo", "heuristic"),
        default="auto",
        help="auto uses YOLO when the model exists, otherwise heuristic foreground",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "models" / "yolov8n.pt",
    )
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--device", default="0", help="Ultralytics device: 0 for CUDA, cpu for CPU")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_local_environment()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    backend = build_backend(args.backend, args.model, args.confidence, args.device)
    engine = VisionEngine(backend)
    worker = FrameWorker(engine)
    worker.start()
    semantic_analyzer = SemanticAnalyzer(SemanticSettings.from_environment())
    memory = SceneMemory(semantic_analyzer)
    semantic_worker = SemanticWorker(semantic_analyzer, memory.set_semantic)
    semantic_worker.start()
    server = PerceptionHTTPServer(
        (args.host, args.port),
        worker,
        backend.name,
        memory,
        semantic_worker,
        semantic_analyzer,
    )
    model_name = backend.name
    try:
        LOGGER.info(
            "ready http://%s:%d fast_model=%s semantic_provider=%s semantic_model=%s",
            args.host,
            args.port,
            model_name,
            semantic_analyzer.settings.provider
            if semantic_analyzer.configured
            else "none",
            semantic_analyzer.settings.model
            if semantic_analyzer.configured
            else "none",
        )
        LOGGER.info(
            "semantic provider=%s model=%s",
            semantic_analyzer.settings.provider,
            semantic_analyzer.settings.model,
        )
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("stopping")
    finally:
        server.server_close()
        semantic_worker.stop()
        worker.stop()
    return 0
