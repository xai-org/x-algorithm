# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import hmac
import ipaddress
import logging
import os
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def default_token_path(port: int) -> str:
    return f"/tmp/status-server-shutdown-token-{port}"


def issue_shutdown_token(port: int, path: str | None = None) -> tuple[str | None, str]:
    token = secrets.token_hex(32)
    token_path = path or default_token_path(port)
    try:
        try:
            os.unlink(token_path)
        except FileNotFoundError:
            pass
        fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(token)
    except OSError as err:
        logger.warning(
            "Failed to write shutdown token file %s (%s); /shutdown will reject all requests.",
            token_path,
            err,
        )
        return None, token_path
    logger.info("Shutdown token written to %s", token_path)
    return token, token_path


def read_shutdown_token(port: int, path: str | None = None) -> str | None:
    try:
        with open(path or default_token_path(port)) as f:
            return f.read().strip()
    except OSError:
        return None


def shutdown_request_authorized(client_ip: str, authorization: str, token: str | None) -> bool:
    try:
        if not ipaddress.ip_address(client_ip).is_loopback:
            return False
    except ValueError:
        return False
    if not token:
        return False
    return hmac.compare_digest(authorization.encode("utf-8"), f"Bearer {token}".encode())


@dataclass(frozen=True, slots=True)
class StatusServerConfig:
    port: int
    ready_event: threading.Event
    on_shutdown: Callable[[], None]
    shutdown_token_path: str | None = None


class StatusServer:
    def __init__(self, *, config: StatusServerConfig):
        self._port: int = config.port
        self._ready_event: threading.Event = config.ready_event
        self._on_shutdown: Callable[[], None] = config.on_shutdown
        self._token_path_override: str | None = config.shutdown_token_path
        self._token_path: str | None = None
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def shutdown_token_path(self) -> str | None:
        return self._token_path

    def start(self) -> None:
        if self._thread is not None:
            return

        ready_event = self._ready_event
        on_shutdown = self._on_shutdown
        state: dict[str, str | None] = {"token": None}

        class Handler(BaseHTTPRequestHandler):
            def _respond(self, status: int, payload: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _handle_shutdown(self) -> None:
                if not shutdown_request_authorized(
                    self.client_address[0],
                    self.headers.get("Authorization", ""),
                    state["token"],
                ):
                    logger.warning(
                        "Rejected unauthorized shutdown request from %s",
                        self.client_address[0],
                    )
                    self._respond(403, b"forbidden\n")
                    return
                ready_event.clear()
                try:
                    on_shutdown()
                except Exception as err:
                    logger.warning("Failed to handle shutdown request: %s", err)
                    self._respond(500, b"shutdown failed\n")
                else:
                    self._respond(200, b"shutdown requested\n")

            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path == "/readyz":
                    if ready_event.is_set():
                        self._respond(200, b"ok\n")
                    else:
                        self._respond(503, b"not ready\n")
                    return
                if path == "/shutdown":
                    self._handle_shutdown()
                    return
                self.send_response(404)
                self.end_headers()

            def do_POST(self) -> None:
                if urlparse(self.path).path == "/shutdown":
                    self._handle_shutdown()
                    return
                self.send_response(404)
                self.end_headers()

            def log_message(self, format, *args):
                return

        try:
            server = ThreadingHTTPServer(("0.0.0.0", self._port), Handler)
        except OSError as err:
            logger.warning("Failed to start status server on port %s: %s", self._port, err)
            return
        bound_port = server.server_address[1]
        token, token_path = issue_shutdown_token(bound_port, self._token_path_override)
        state["token"] = token
        self._token_path = token_path
        thread = threading.Thread(
            target=server.serve_forever, name="status_http_server", daemon=True
        )
        thread.start()
        self._server = server
        self._thread = thread

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._token_path is not None:
            try:
                os.unlink(self._token_path)
            except OSError:
                pass
            self._token_path = None
        self._thread = None
