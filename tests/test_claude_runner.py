"""Tests for the Claude CLI subprocess wrapper and verdict/confidence extractors."""
from __future__ import annotations

import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from debater.claude_runner import (
    debate_log,
    extract_confidence_with_llm,
    extract_verdict_with_llm,
    run_agent,
    run_claude_cmd,
)
from debater.config import AgentError


def _mock_popen(stdout="ok", stderr="", returncode=0):
    """Build a MagicMock that mimics subprocess.Popen() for happy-path tests."""
    proc = MagicMock()
    proc.communicate.return_value = (stdout, stderr)
    proc.returncode = returncode
    return proc


class TestRunClaudeCmd:
    def test_happy_path_releases_semaphore(self, registry):
        # Pre-acquire one slot so we can detect whether release happens.
        initial_free = 2
        with patch("debater.claude_runner.subprocess.Popen",
                   return_value=_mock_popen(stdout="hello")):
            result = run_claude_cmd(registry, ["claude", "--help"])
        assert result.returncode == 0
        assert result.stdout == "hello"
        # All semaphore slots should be free again
        for _ in range(initial_free):
            assert registry.claude_semaphore.acquire(blocking=False)

    def test_releases_semaphore_on_timeout(self, registry):
        proc = MagicMock()
        proc.communicate.side_effect = subprocess.TimeoutExpired(cmd=["x"], timeout=1)
        with patch("debater.claude_runner.subprocess.Popen", return_value=proc):
            with pytest.raises(subprocess.TimeoutExpired):
                run_claude_cmd(registry, ["claude"], timeout=1)
        # Slot was acquired then released even though the call raised
        for _ in range(2):
            assert registry.claude_semaphore.acquire(blocking=False)

    def test_releases_semaphore_on_subprocess_error(self, registry):
        with patch("debater.claude_runner.subprocess.Popen", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                run_claude_cmd(registry, ["claude"])
        for _ in range(2):
            assert registry.claude_semaphore.acquire(blocking=False)

    def test_registers_and_unregisters_process(self, registry):
        with patch("debater.claude_runner.subprocess.Popen",
                   return_value=_mock_popen()) as popen:
            run_claude_cmd(registry, ["claude"], debate_id="dbg")
        assert "dbg" not in registry.active_processes
        popen.assert_called_once()

    def test_no_process_tracking_without_debate_id(self, registry):
        with patch("debater.claude_runner.subprocess.Popen", return_value=_mock_popen()):
            run_claude_cmd(registry, ["claude"])
        # Without debate_id we don't touch the dict at all
        assert registry.active_processes == {}


class TestRunAgent:
    def test_returns_stdout_on_success(self, registry):
        with patch("debater.claude_runner.subprocess.Popen",
                   return_value=_mock_popen(stdout="defender response")):
            out = run_agent(registry, "defender", "prompt", "claude-haiku")
        assert out == "defender response"

    def test_raises_agent_error_on_nonzero_exit(self, registry):
        with patch("debater.claude_runner.subprocess.Popen",
                   return_value=_mock_popen(stderr="boom", returncode=1)):
            with pytest.raises(AgentError, match="exited with code 1"):
                run_agent(registry, "defender", "prompt", "claude-haiku")

    def test_raises_agent_error_on_empty_output(self, registry):
        with patch("debater.claude_runner.subprocess.Popen",
                   return_value=_mock_popen(stdout="   ")):
            with pytest.raises(AgentError, match="empty output"):
                run_agent(registry, "defender", "prompt", "claude-haiku")

    def test_raises_agent_error_when_claude_missing(self, registry):
        with patch("debater.claude_runner.subprocess.Popen", side_effect=FileNotFoundError):
            with pytest.raises(AgentError, match="claude CLI not found"):
                run_agent(registry, "defender", "prompt", "claude-haiku")

    def test_includes_effort_flag(self, registry):
        with patch("debater.claude_runner.subprocess.Popen",
                   return_value=_mock_popen()) as popen:
            run_agent(registry, "defender", "prompt", "claude-haiku", effort="high")
        cmd = popen.call_args[0][0]
        assert "--effort" in cmd
        assert cmd[cmd.index("--effort") + 1] == "high"

    def test_includes_allowed_tools(self, registry):
        with patch("debater.claude_runner.subprocess.Popen",
                   return_value=_mock_popen()) as popen:
            run_agent(registry, "researcher", "p", "m", allowed_tools="WebSearch")
        cmd = popen.call_args[0][0]
        assert "--allowedTools" in cmd
        assert cmd[cmd.index("--allowedTools") + 1] == "WebSearch"


class TestDebateLog:
    def test_writes_timestamped_entry(self, tmp_path):
        d = str(tmp_path)
        debate_log(d, "AGENT_PROMPT [defender]", "hello world")
        content = (tmp_path / "debug.log").read_text()
        assert "AGENT_PROMPT [defender]" in content
        assert "hello world" in content
        # Timestamp pattern
        assert "T" in content and "Z" in content

    def test_noop_when_dir_is_none(self):
        # Must not raise
        debate_log(None, "X", "Y")

    def test_swallows_write_errors(self, tmp_path):
        # Point at a non-writable location
        bad = str(tmp_path / "nope" / "subdir")
        # Function swallows all exceptions and returns silently
        debate_log(bad, "X", "Y")
        # File should not exist
        assert not os.path.exists(os.path.join(bad, "debug.log"))


class TestVerdictAndConfidenceExtractors:
    def test_verdict_exact_match(self, registry, mock_run_agent):
        mock_run_agent.set(lambda **kw: "REFUTED")
        assert extract_verdict_with_llm(registry, "summary") == "REFUTED"

    def test_verdict_substring_match(self, registry, mock_run_agent):
        mock_run_agent.set(lambda **kw: "After analysis: STRONGLY REFUTED is the verdict")
        assert extract_verdict_with_llm(registry, "summary") == "STRONGLY REFUTED"

    def test_verdict_falls_back_to_inconclusive(self, registry, mock_run_agent):
        mock_run_agent.set(lambda **kw: "nonsense response")
        assert extract_verdict_with_llm(registry, "summary") == "INCONCLUSIVE"

    def test_verdict_handles_exception(self, registry, monkeypatch):
        def boom(*a, **kw):
            raise RuntimeError()
        monkeypatch.setattr("debater.claude_runner.run_agent", boom)
        assert extract_verdict_with_llm(registry, "summary") == "INCONCLUSIVE"

    def test_confidence_extracted_from_digits(self, registry, mock_run_agent):
        mock_run_agent.set(lambda **kw: "85")
        assert extract_confidence_with_llm(registry, "x") == 85

    def test_confidence_clamps_above_100(self, registry, mock_run_agent):
        mock_run_agent.set(lambda **kw: "200")
        assert extract_confidence_with_llm(registry, "x") == 100

    def test_confidence_default_70_when_no_digits(self, registry, mock_run_agent):
        mock_run_agent.set(lambda **kw: "no number")
        assert extract_confidence_with_llm(registry, "x") == 70

    def test_confidence_default_on_exception(self, registry, monkeypatch):
        def boom(*a, **kw):
            raise RuntimeError()
        monkeypatch.setattr("debater.claude_runner.run_agent", boom)
        assert extract_confidence_with_llm(registry, "x") == 70
