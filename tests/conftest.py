"""Shared pytest fixtures.

The Claude CLI is never invoked from tests: `mock_run_agent` patches every
`run_agent` import-site so callers receive canned responses keyed by agent name.
"""
from __future__ import annotations

from collections.abc import Callable

import pytest

from debater.registry import DebateRegistry


@pytest.fixture
def tmp_debates_dir(tmp_path):
    """Fresh debates directory for a single test."""
    d = tmp_path / "debates"
    d.mkdir()
    return str(d)


@pytest.fixture
def registry(tmp_debates_dir):
    """A DebateRegistry rooted in a tmp debates dir, single-slot semaphore."""
    return DebateRegistry(debates_dir=tmp_debates_dir, max_concurrent_claude=2)


@pytest.fixture
def mock_run_agent(monkeypatch):
    """Patch run_agent in every module that imported it.

    Usage:
        def test_foo(mock_run_agent):
            mock_run_agent.set(lambda agent_name, prompt, **kw: f"reply from {agent_name}")
            # ... call code that ends up invoking run_agent ...
            assert mock_run_agent.calls[0]["agent_name"] == "defender"
    """

    class Handle:
        def __init__(self):
            self._responder: Callable[..., str] = lambda **kw: "ok"
            self.calls: list[dict] = []

        def set(self, fn: Callable[..., str]) -> None:
            self._responder = fn

        def fake(self, *args, **kwargs):
            # The real signature: run_agent(registry, agent_name, prompt, model, ...)
            # We accept the registry positionally and pass everything else through.
            if len(args) >= 2:
                agent_name = args[1]
                prompt = args[2] if len(args) >= 3 else kwargs.get("prompt", "")
            else:
                agent_name = kwargs.get("agent_name", "?")
                prompt = kwargs.get("prompt", "")
            record = {"agent_name": agent_name, "prompt": prompt, "kwargs": kwargs}
            self.calls.append(record)
            return self._responder(agent_name=agent_name, prompt=prompt, **kwargs)

    handle = Handle()

    # Patch every module that imported run_agent.
    import debater.claude_runner as _cr
    import debater.orchestrator as _orch
    import debater.research as _res

    monkeypatch.setattr(_cr, "run_agent", handle.fake)
    monkeypatch.setattr(_orch, "run_agent", handle.fake)
    monkeypatch.setattr(_res, "run_agent", handle.fake)

    return handle
