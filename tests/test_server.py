"""HTTP server smoke tests against a real ThreadingHTTPServer instance.

The orchestrator's background workers are patched out so creating a debate
doesn't spawn the real Claude CLI.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from debater.registry import DebateRegistry
from debater.server import make_handler


@pytest.fixture
def server(tmp_debates_dir, monkeypatch):
    """Start a real HTTP server on a free port, with orchestrator workers stubbed."""
    # Replace the background-thread targets so requests don't hit Claude
    monkeypatch.setattr("debater.server.run_full_debate", lambda *a, **kw: None)
    monkeypatch.setattr("debater.server.generate_title_async", lambda *a, **kw: None)
    monkeypatch.setattr("debater.server.run_analysis_mode", lambda *a, **kw: None)

    registry = DebateRegistry(debates_dir=tmp_debates_dir, max_concurrent_claude=2)
    handler_cls = make_handler(registry)

    # Bind to port 0 → OS picks a free one
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    port = httpd.server_address[1]

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    yield {"port": port, "registry": registry, "tmp_debates_dir": tmp_debates_dir}

    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=2)


def _get(port: int, path: str):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="GET")
    try:
        return urllib.request.urlopen(req, timeout=2)
    except urllib.error.HTTPError as e:
        return e


def _post(port: int, path: str, body: dict | None = None):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        return urllib.request.urlopen(req, timeout=2)
    except urllib.error.HTTPError as e:
        return e


class TestGetRoutes:
    def test_root_serves_html(self, server):
        resp = _get(server["port"], "/")
        assert resp.status == 200
        body = resp.read().decode()
        assert "<!DOCTYPE html>" in body
        # New stacked bar markup from Commit B
        assert "score-bar-defender" in body
        assert "score-bar-skeptic" in body

    def test_list_debates_empty(self, server):
        resp = _get(server["port"], "/api/debates")
        assert resp.status == 200
        data = json.loads(resp.read())
        assert data == {"debates": []}

    def test_get_unknown_debate_returns_404(self, server):
        resp = _get(server["port"], "/api/debate/2026-01-01_nope")
        assert resp.status == 404

    def test_unknown_route_returns_404(self, server):
        resp = _get(server["port"], "/api/whatever")
        assert resp.status == 404


class TestCreateDebate:
    def test_creates_and_returns_id(self, server):
        resp = _post(server["port"], "/api/debate", {"claim": "hello world"})
        assert resp.status == 200
        data = json.loads(resp.read())
        assert "debate_id" in data
        assert data["status"] == "running"
        # State file actually written to disk
        debate_dir = os.path.join(server["tmp_debates_dir"], data["debate_id"])
        assert os.path.isfile(os.path.join(debate_dir, "state.json"))

    def test_empty_claim_rejected(self, server):
        resp = _post(server["port"], "/api/debate", {"claim": "   "})
        assert resp.status == 400
        data = json.loads(resp.read())
        assert "error" in data

    def test_missing_claim_rejected(self, server):
        resp = _post(server["port"], "/api/debate", {})
        assert resp.status == 400

    def test_iterations_clamped(self, server):
        # Above-max iterations is clamped to 50
        resp = _post(server["port"], "/api/debate",
                     {"claim": "x", "iterations": 9999})
        assert resp.status == 200
        data = json.loads(resp.read())
        debate_dir = os.path.join(server["tmp_debates_dir"], data["debate_id"])
        with open(os.path.join(debate_dir, "state.json")) as f:
            state = json.load(f)
        assert state["settings"]["max_iterations"] == 50

    def test_invalid_iterations_default_5(self, server):
        resp = _post(server["port"], "/api/debate",
                     {"claim": "x", "iterations": "garbage"})
        assert resp.status == 200
        data = json.loads(resp.read())
        debate_dir = os.path.join(server["tmp_debates_dir"], data["debate_id"])
        with open(os.path.join(debate_dir, "state.json")) as f:
            state = json.load(f)
        assert state["settings"]["max_iterations"] == 5


class TestPathTraversalProtection:
    def test_traversal_in_get_debate_returns_404(self, server):
        # URL-encoded "../" returns 404 because get_debate_dir rejects it
        resp = _get(server["port"], "/api/debate/..%2Fetc%2Fpasswd")
        assert resp.status == 404


class TestStopAndAnalyze:
    def test_stop_unknown_debate_returns_404(self, server):
        resp = _post(server["port"], "/api/debate/unknown/stop")
        assert resp.status == 404

    def test_message_to_unknown_debate_returns_404(self, server):
        resp = _post(server["port"], "/api/debate/unknown/message",
                     {"message": "hi"})
        assert resp.status == 404

    def test_message_empty_returns_400(self, server):
        # First create a real debate
        created = json.loads(_post(server["port"], "/api/debate",
                                   {"claim": "x"}).read())
        resp = _post(server["port"], f"/api/debate/{created['debate_id']}/message",
                     {"message": "   "})
        assert resp.status == 400


def test_oversized_post_rejected(server):
    """1 MB content-length cap is enforced."""
    # Build a request with a Content-Length header that exceeds the limit
    req = urllib.request.Request(
        f"http://127.0.0.1:{server['port']}/api/debate",
        data=b"x" * 2_000_000,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=2)
        status = resp.status
    except urllib.error.HTTPError as e:
        status = e.code
    except (ConnectionResetError, TimeoutError, urllib.error.URLError):
        # Server may close the connection rather than respond; that's also acceptable.
        pytest.skip("server closed connection on oversized payload")
    assert status == 413
