"""Tests for the DebateRegistry: locks, thread bookkeeping, process tracking."""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

from debater.registry import DebateRegistry


class TestStateLock:
    def test_returns_same_lock_for_same_id(self, registry):
        lock1 = registry.state_lock("a")
        lock2 = registry.state_lock("a")
        assert lock1 is lock2

    def test_returns_different_locks_for_different_ids(self, registry):
        assert registry.state_lock("a") is not registry.state_lock("b")

    def test_lock_is_a_threading_lock(self, registry):
        lock = registry.state_lock("a")
        # threading.Lock returns a custom _thread.lock object that supports the
        # context-manager protocol; check usability rather than exact type.
        with lock:
            pass


class TestDebateRegistration:
    def test_register_and_query(self, registry):
        thread = threading.Thread(target=lambda: None)
        event = threading.Event()
        registry.register_debate("foo", thread, event)
        assert registry.is_active("foo")
        assert registry.get_event("foo") is event

    def test_unregister_removes(self, registry):
        registry.register_debate("foo", threading.Thread(target=lambda: None), threading.Event())
        registry.unregister_debate("foo")
        assert not registry.is_active("foo")
        assert registry.get_event("foo") is None

    def test_unregister_unknown_is_safe(self, registry):
        registry.unregister_debate("never-registered")  # must not raise

    def test_is_active_false_for_unknown(self, registry):
        assert not registry.is_active("nope")


class TestProcessTracking:
    def test_register_and_unregister(self, registry):
        proc = MagicMock()
        registry.register_process("foo", proc)
        # Direct dict inspection
        assert registry.active_processes["foo"] is proc
        registry.unregister_process("foo")
        assert "foo" not in registry.active_processes

    def test_kill_process_returns_true_when_present(self, registry):
        proc = MagicMock()
        registry.register_process("foo", proc)
        assert registry.kill_process("foo") is True
        proc.kill.assert_called_once()

    def test_kill_process_returns_false_when_missing(self, registry):
        assert registry.kill_process("nope") is False

    def test_kill_process_handles_oserror(self, registry):
        proc = MagicMock()
        proc.kill.side_effect = OSError("process already dead")
        registry.register_process("foo", proc)
        # Must not raise; returns False
        assert registry.kill_process("foo") is False


class TestSignalAllStop:
    def test_sets_all_events_and_kills_all_procs(self, registry):
        event1, event2 = threading.Event(), threading.Event()
        thread_dummy = threading.Thread(target=lambda: None)
        registry.register_debate("a", thread_dummy, event1)
        registry.register_debate("b", thread_dummy, event2)
        proc1, proc2 = MagicMock(), MagicMock()
        registry.register_process("a", proc1)
        registry.register_process("b", proc2)

        registry.signal_all_stop()

        assert event1.is_set()
        assert event2.is_set()
        proc1.kill.assert_called_once()
        proc2.kill.assert_called_once()

    def test_swallows_oserror_during_kill(self, registry):
        proc = MagicMock()
        proc.kill.side_effect = OSError("zombie")
        registry.register_process("a", proc)
        # Must not raise
        registry.signal_all_stop()


class TestConcurrencySemantics:
    def test_state_lock_concurrent_calls_return_same_lock(self, registry):
        # 20 threads request state_lock for the same id; all must get the same object.
        results: list[object] = []
        barrier = threading.Barrier(20)

        def worker():
            barrier.wait()
            results.append(registry.state_lock("shared"))

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        assert all(r is results[0] for r in results)


def test_registry_constructor_defaults():
    """Smoke test: a fresh registry has the right initial state."""
    r = DebateRegistry(debates_dir="/tmp/x", max_concurrent_claude=3)
    assert r.debates_dir == "/tmp/x"
    assert r.active_debates == {}
    assert r.debate_events == {}
    assert r.active_processes == {}
    assert r.state_locks == {}
    assert r.server is None
    assert r.shutdown_flag is False
    # Semaphore can be acquired up to its limit
    assert r.claude_semaphore.acquire(blocking=False)
    assert r.claude_semaphore.acquire(blocking=False)
    assert r.claude_semaphore.acquire(blocking=False)
    assert not r.claude_semaphore.acquire(blocking=False)  # 4th fails
