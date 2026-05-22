"""Runtime registry that owns shared concurrency primitives and bookkeeping.

Lock ordering invariant: `lock` (the registry-wide lock) is always acquired
BEFORE any per-debate `state_lock(id)`. Never invert this ordering.
"""
from __future__ import annotations

import subprocess
import threading
from http.server import ThreadingHTTPServer


class DebateRegistry:
    def __init__(self, debates_dir: str, max_concurrent_claude: int):
        self.debates_dir = debates_dir
        self.claude_semaphore = threading.Semaphore(max_concurrent_claude)
        self.active_debates: dict[str, threading.Thread] = {}
        self.debate_events: dict[str, threading.Event] = {}
        self.active_processes: dict[str, subprocess.Popen] = {}
        self.state_locks: dict[str, threading.Lock] = {}
        self.lock = threading.Lock()
        self.server: ThreadingHTTPServer | None = None
        self.shutdown_flag = False

    def state_lock(self, debate_id: str) -> threading.Lock:
        with self.lock:
            if debate_id not in self.state_locks:
                self.state_locks[debate_id] = threading.Lock()
            return self.state_locks[debate_id]

    def register_debate(self, debate_id: str, thread: threading.Thread,
                        event: threading.Event) -> None:
        with self.lock:
            self.active_debates[debate_id] = thread
            self.debate_events[debate_id] = event

    def unregister_debate(self, debate_id: str) -> None:
        with self.lock:
            self.active_debates.pop(debate_id, None)
            self.debate_events.pop(debate_id, None)

    def is_active(self, debate_id: str) -> bool:
        with self.lock:
            return debate_id in self.active_debates

    def get_event(self, debate_id: str) -> threading.Event | None:
        with self.lock:
            return self.debate_events.get(debate_id)

    def register_process(self, debate_id: str, proc: subprocess.Popen) -> None:
        with self.lock:
            self.active_processes[debate_id] = proc

    def unregister_process(self, debate_id: str) -> None:
        with self.lock:
            self.active_processes.pop(debate_id, None)

    def kill_process(self, debate_id: str) -> bool:
        with self.lock:
            proc = self.active_processes.get(debate_id)
        if proc:
            try:
                proc.kill()
                return True
            except OSError:
                return False
        return False

    def signal_all_stop(self) -> None:
        """Signal all running debates to stop and kill all active subprocesses."""
        with self.lock:
            events = list(self.debate_events.values())
            procs = list(self.active_processes.values())
        for event in events:
            event.set()
        for proc in procs:
            try:
                proc.kill()
            except OSError:
                pass

    def join_all(self, timeout: float = 5.0) -> None:
        with self.lock:
            threads = list(self.active_debates.values())
        for thread in threads:
            thread.join(timeout=timeout)
