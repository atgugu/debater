"""HTTP server: routes + handler factory."""
from __future__ import annotations

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler

from debater.orchestrator import generate_title_async, run_analysis_mode, run_full_debate
from debater.registry import DebateRegistry
from debater.state import (
    append_md,
    create_debate,
    get_debate_dir,
    list_debates,
    load_state,
    now_iso,
    read_file,
    save_state,
)

log = logging.getLogger("debater")

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "static")


def make_handler(registry: DebateRegistry) -> type[BaseHTTPRequestHandler]:
    """Factory: returns a Handler class bound to this registry."""

    class DebaterHandler(BaseHTTPRequestHandler):
        _registry = registry

        def do_GET(self):
            if self.path == "/":
                self._serve_html()
            elif self.path == "/api/debates":
                self._handle_list_debates()
            elif self.path.startswith("/api/debate/"):
                parts = self.path.rstrip("/").split("/")
                if len(parts) == 4:
                    self._handle_get_debate(parts[3])
                else:
                    self._send_error(404, "Not found")
            else:
                self._send_error(404, "Not found")

        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 1_000_000:
                self._send_error(413, "Request too large")
                return
            body = {}
            if content_length > 0:
                raw = self.rfile.read(content_length)
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError:
                    self._send_error(400, "Invalid JSON")
                    return

            if self.path == "/api/debate":
                self._handle_create_debate(body)
            elif self.path.startswith("/api/debate/") and self.path.endswith("/message"):
                parts = self.path.rstrip("/").split("/")
                self._handle_message(parts[3], body)
            elif self.path.startswith("/api/debate/") and self.path.endswith("/resume"):
                parts = self.path.rstrip("/").split("/")
                self._handle_resume(parts[3], body)
            elif self.path.startswith("/api/debate/") and self.path.endswith("/stop"):
                parts = self.path.rstrip("/").split("/")
                self._handle_stop(parts[3])
            elif self.path.startswith("/api/debate/") and self.path.endswith("/analyze"):
                parts = self.path.rstrip("/").split("/")
                self._handle_analyze(parts[3])
            else:
                self._send_error(404, "Not found")

        def _serve_html(self):
            html_path = os.path.join(STATIC_DIR, "index.html")
            try:
                with open(html_path, encoding="utf-8") as f:
                    content = f.read()
            except FileNotFoundError:
                self._send_error(500, "index.html not found")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(content.encode())

        def _send_json(self, data, status=200):
            body = json.dumps(data, ensure_ascii=False)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body.encode())

        def _send_error(self, status, message):
            self._send_json({"error": message}, status)

        def _handle_list_debates(self):
            self._send_json({"debates": list_debates(self._registry.debates_dir)})

        def _handle_get_debate(self, debate_id):
            debate_dir = get_debate_dir(self._registry.debates_dir, debate_id)
            state = load_state(debate_dir)
            if not state:
                self._send_error(404, "Debate not found")
                return

            data = {
                "state": state,
                "defender_md": read_file(os.path.join(debate_dir, "defender.md")),
                "skeptic_md": read_file(os.path.join(debate_dir, "skeptic.md")),
                "evaluator_md": read_file(os.path.join(debate_dir, "evaluator.md")),
                "research_md": read_file(os.path.join(debate_dir, "research.md")),
                "summary_md": read_file(os.path.join(debate_dir, "summary.md")),
                "user_messages_md": read_file(os.path.join(debate_dir, "user_messages.md")),
            }
            self._send_json(data)

        def _handle_create_debate(self, body):
            claim = body.get("claim", "").strip()
            if not claim:
                self._send_error(400, "Claim is required")
                return

            try:
                iterations = max(1, min(50, int(body.get("iterations", 5))))
            except (TypeError, ValueError):
                iterations = 5
            settings = body.get("settings", {})
            settings["max_iterations"] = iterations

            prior_ids = body.get("prior_debate_ids", [])
            prior_debates = []
            for pid in prior_ids:
                prior_dir = get_debate_dir(self._registry.debates_dir, pid)
                prior_state = load_state(prior_dir)
                if prior_state and prior_state.get("summary"):
                    prior_debates.append({
                        "id": pid,
                        "claim": prior_state.get("claim", ""),
                        "verdict": prior_state.get("verdict", "INCONCLUSIVE"),
                        "summary": prior_state.get("summary", ""),
                    })
            if prior_debates:
                settings["prior_debates"] = prior_debates
                log.info("Injecting %d prior debate conclusion(s)", len(prior_debates))

            debate_id = create_debate(self._registry.debates_dir, claim, settings)
            debate_dir = get_debate_dir(self._registry.debates_dir, debate_id)

            registry_ref = self._registry
            threading.Thread(target=generate_title_async,
                             args=(registry_ref, debate_dir),
                             daemon=True).start()

            shutdown_event = threading.Event()
            thread = threading.Thread(
                target=run_full_debate,
                args=(registry_ref, debate_dir, shutdown_event),
                daemon=True,
            )

            registry_ref.register_debate(debate_id, thread, shutdown_event)

            thread.start()
            self._send_json({"debate_id": debate_id, "status": "running"})

        def _handle_message(self, debate_id, body):
            debate_dir = get_debate_dir(self._registry.debates_dir, debate_id)
            if debate_dir is None:
                self._send_error(400, "Invalid debate ID")
                return
            lock = self._registry.state_lock(debate_id)

            with lock:
                state = load_state(debate_dir)
                if not state:
                    self._send_error(404, "Debate not found")
                    return

                message = body.get("message", "").strip()
                if not message:
                    self._send_error(400, "Message is required")
                    return

                state.setdefault("user_messages", []).append({
                    "message": message,
                    "timestamp": now_iso(),
                    "during_round": state.get("current_round", 0),
                    "during_agent": state.get("active_agent") or "between rounds",
                })
                save_state(debate_dir, state)

            append_md(
                os.path.join(debate_dir, "user_messages.md"),
                f"Interjection at {now_iso()}",
                message,
            )

            self._send_json({"status": "ok", "message_count": len(state["user_messages"])})

        def _handle_resume(self, debate_id, body):
            debate_dir = get_debate_dir(self._registry.debates_dir, debate_id)
            if debate_dir is None:
                self._send_error(400, "Invalid debate ID")
                return

            if self._registry.is_active(debate_id):
                self._send_error(409, "Debate is already running")
                return

            lock = self._registry.state_lock(debate_id)
            with lock:
                state = load_state(debate_dir)
                if not state:
                    self._send_error(404, "Debate not found")
                    return

                if state["status"] == "running":
                    self._send_error(409, "Debate is already running")
                    return

                additional = body.get("additional_iterations", 0)
                if additional > 0:
                    state["settings"]["max_iterations"] += additional
                state["status"] = "pending"
                state["early_stopped"] = False
                state["verdict"] = None
                state["summary"] = None
                save_state(debate_dir, state)

            shutdown_event = threading.Event()
            registry_ref = self._registry
            thread = threading.Thread(
                target=run_full_debate,
                args=(registry_ref, debate_dir, shutdown_event),
                daemon=True,
            )

            registry_ref.register_debate(debate_id, thread, shutdown_event)

            thread.start()
            self._send_json({
                "status": "running",
                "max_iterations": state["settings"]["max_iterations"],
            })

        def _handle_analyze(self, debate_id):
            debate_dir = get_debate_dir(self._registry.debates_dir, debate_id)
            state = load_state(debate_dir)
            if not state:
                self._send_error(404, "Debate not found")
                return
            if state.get("status") != "refused":
                self._send_error(400, "Debate is not in refused state")
                return

            registry_ref = self._registry
            thread = threading.Thread(
                target=run_analysis_mode,
                args=(registry_ref, debate_dir, state),
                daemon=True,
            )
            thread.start()
            self._send_json({"status": "analyzing"})

        def _handle_stop(self, debate_id):
            event = self._registry.get_event(debate_id)
            if event:
                event.set()
                self._registry.kill_process(debate_id)
                self._send_json({"status": "stopping"})
            else:
                self._send_error(404, "No active debate with that ID")

        def log_message(self, format, *args):
            pass

    return DebaterHandler
