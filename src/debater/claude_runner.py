"""Wrapper around the Claude CLI subprocess plus small LLM-call helpers."""
from __future__ import annotations

import logging
import os
import subprocess

from debater.config import VERDICT_SCALE, AgentError
from debater.registry import DebateRegistry
from debater.state import now_iso

log = logging.getLogger("debater")


def debate_log(debate_dir: str | None, category: str, content: str) -> None:
    """Append a timestamped entry to the debate's debug.log file."""
    if not debate_dir:
        return
    try:
        path = os.path.join(debate_dir, "debug.log")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n[{now_iso()}] {category}\n{'='*80}\n{content}\n")
    except Exception:
        pass  # Never let debug logging break the debate


def run_claude_cmd(registry: DebateRegistry, cmd: list[str], timeout: int = 120,
                   debate_id: str | None = None) -> subprocess.CompletedProcess:
    """Run a Claude CLI command with concurrency throttling. Killable via debate_id."""
    registry.claude_semaphore.acquire()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if debate_id:
            registry.register_process(debate_id, proc)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
        finally:
            if debate_id:
                registry.unregister_process(debate_id)
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout=stdout, stderr=stderr)
    finally:
        registry.claude_semaphore.release()


def run_agent(registry: DebateRegistry, agent_name: str, prompt: str, model: str,
              effort: str | None = None, allowed_tools: str | None = None,
              timeout: int = 600, debate_id: str | None = None,
              debate_dir: str | None = None) -> str:
    """Run a single agent via Claude CLI. Returns stdout text."""
    cmd = ["claude", "--model", model, "--output-format", "text", "-p", prompt]
    if effort:
        cmd.extend(["--effort", effort])
    if allowed_tools:
        cmd.extend(["--allowedTools", allowed_tools])

    log.info("[%s] Running with model=%s effort=%s ...", agent_name, model, effort or "default")
    debate_log(debate_dir, f"AGENT_PROMPT [{agent_name}]",
               f"Model: {model}\nEffort: {effort or 'default'}\nTools: {allowed_tools or 'none'}\n\n{prompt}")
    try:
        result = run_claude_cmd(registry, cmd, timeout=timeout, debate_id=debate_id)
    except FileNotFoundError:
        raise AgentError("claude CLI not found — is it installed and on PATH?")
    except subprocess.TimeoutExpired:
        debate_log(debate_dir, f"AGENT_ERROR [{agent_name}]", f"Timed out after {timeout}s")
        raise AgentError(f"Claude CLI timed out after {timeout}s")

    log.debug("[%s] returned %d chars (exit %d)", agent_name, len(result.stdout), result.returncode)
    if result.stderr and result.stderr.strip():
        log.debug("[%s] stderr: %s", agent_name, result.stderr.strip()[:500])

    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else "unknown error"
        debate_log(debate_dir, f"AGENT_ERROR [{agent_name}]",
                   f"Exit code: {result.returncode}\nStderr: {stderr[:1000]}")
        raise AgentError(f"CLI exited with code {result.returncode}: {stderr[:500]}")

    output = result.stdout.strip()
    if not output:
        debate_log(debate_dir, f"AGENT_ERROR [{agent_name}]", "Empty output")
        raise AgentError("CLI returned empty output")

    debate_log(debate_dir, f"AGENT_RESPONSE [{agent_name}]",
               f"Length: {len(output)} chars\n\n{output}")
    return output


def extract_confidence_with_llm(registry: DebateRegistry, text: str,
                                debate_id: str | None = None,
                                debate_dir: str | None = None) -> int:
    """Extract the confidence level from summary text as a percentage using a Haiku call."""
    prompt = (
        "Read the debate summary below. Find the confidence level stated in the verdict section.\n\n"
        "Return ONLY a number between 0 and 100 representing the confidence percentage.\n"
        "Examples: '99.9%' → 99, 'High (80%+)' → 80, 'Moderate' → 60, 'Low' → 30\n"
        "If no confidence level is stated, return: 70\n\n"
        f"---\n\n{text[-5000:]}"
    )
    try:
        result = run_agent(
            registry, "confidence_extractor", prompt,
            "claude-haiku-4-5-20251001",
            timeout=30, debate_id=debate_id, debate_dir=debate_dir,
        )
        num = "".join(c for c in result.strip() if c.isdigit())
        if num:
            return max(0, min(100, int(num)))
        return 70
    except Exception:
        return 70


def extract_verdict_with_llm(registry: DebateRegistry, text: str,
                             debate_id: str | None = None,
                             debate_dir: str | None = None) -> str:
    """Semantically extract the final verdict from summary text using a Haiku call."""
    verdict_options = ", ".join(VERDICT_SCALE)
    prompt = (
        f"Read the debate summary below. Identify the FINAL overall verdict about the claim "
        f"being debated. The verdict appears in a section labeled '## Verdict' near the end.\n\n"
        f"Return ONLY one of these exact values (all caps, no other text):\n"
        f"{verdict_options}\n\n"
        f"If you cannot determine a clear verdict, return: INCONCLUSIVE\n\n"
        f"---\n\n{text}"
    )
    try:
        result = run_agent(
            registry, "verdict_extractor", prompt,
            "claude-haiku-4-5-20251001",
            timeout=30, debate_id=debate_id, debate_dir=debate_dir,
        )
        result = result.strip().upper()
        for v in VERDICT_SCALE:
            if result == v:
                return v
        for v in VERDICT_SCALE:
            if v in result:
                return v
        return "INCONCLUSIVE"
    except Exception:
        return "INCONCLUSIVE"
