"""Research request extraction and execution."""
from __future__ import annotations

import logging
import os
import re

from debater.claude_runner import debate_log, run_agent
from debater.config import AgentError
from debater.prompts import build_researcher_prompt
from debater.registry import DebateRegistry
from debater.state import append_md, load_state, now_iso, save_state

log = logging.getLogger("debater")


def extract_research_requests(text: str) -> list[str]:
    """Extract <research_request>...</research_request> tags from response."""
    return re.findall(r"<research_request>(.*?)</research_request>", text, re.DOTALL)


def run_researcher(registry: DebateRegistry, query: str, claim: str, requesting_agent: str,
                   model: str, timeout: int = 120, debate_id: str | None = None,
                   debate_dir: str | None = None) -> str:
    """Run the online researcher agent with WebSearch."""
    prompt = build_researcher_prompt(query, claim, requesting_agent)
    return run_agent(registry, "researcher", prompt, model, allowed_tools="WebSearch",
                     timeout=timeout, debate_id=debate_id, debate_dir=debate_dir)


def run_agent_with_research(registry: DebateRegistry, agent_name: str, prompt: str,
                            model: str, effort: str | None, state: dict, debate_dir: str,
                            round_data: dict) -> str | None:
    """
    Run an agent. If the response contains <research_request> tags:
    pause, research, inject results, re-run agent.
    Returns final response text.
    """
    debate_id = state["debate_id"]
    try:
        response = run_agent(registry, agent_name, prompt, model, effort=effort,
                             debate_id=debate_id, debate_dir=debate_dir)
    except AgentError as e:
        if "code -9" in str(e) or "code -15" in str(e):
            log.info("[%s] Killed by stop signal", agent_name)
            return None
        log.error("[%s] Agent failed: %s", agent_name, e)
        return f"**[Agent error: {e}]**"

    queries = extract_research_requests(response)
    if not queries:
        return response

    log.info("[%s] Requested %d research queries", agent_name, len(queries))
    debate_log(debate_dir, f"RESEARCH_REQUEST [{agent_name}]", f"Queries: {queries}")
    researcher_model = state["settings"]["models"]["researcher"]
    research_results = []

    lock = registry.state_lock(state["debate_id"])
    with lock:
        current_state = load_state(debate_dir)
        current_state["agent_status"] = "researching"
        save_state(debate_dir, current_state)

    max_queries = 10 if agent_name == "fact_checker" else 3
    for query in queries[:max_queries]:
        query = query.strip()
        if not query:
            continue
        log.info("[%s] Researching: %s", agent_name, query[:80])
        try:
            result = run_researcher(registry, query, state["claim"], agent_name,
                                    researcher_model, debate_id=debate_id, debate_dir=debate_dir)
        except AgentError as e:
            result = f"[Research failed for this query. No results available. Error: {e}]"
            log.warning("[%s] Research failed for '%s': %s", agent_name, query[:50], e)

        research_entry = {
            "requested_by": agent_name,
            "query": query,
            "results": result,
            "timestamp": now_iso(),
        }
        research_results.append(research_entry)
        round_data.setdefault("research", []).append(research_entry)

        research_md_path = os.path.join(debate_dir, "research.md")
        append_md(
            research_md_path,
            f"Research for {agent_name.title()} — Round {round_data['round']}",
            f"**Query:** {query}\n\n{result}",
        )

    research_block = "\n\nRESEARCH RESULTS:\n"
    research_block += "The following research was conducted based on your request(s):\n\n"
    for res in research_results:
        research_block += f'Query: "{res["query"]}"\nResults:\n{res["results"]}\n\n'
    research_block += "Now provide your COMPLETE argument for this round, incorporating these research findings. Do NOT output any more <research_request> tags."

    augmented_prompt = prompt + research_block

    with lock:
        current_state = load_state(debate_dir)
        current_state["agent_status"] = "thinking"
        save_state(debate_dir, current_state)

    try:
        response = run_agent(registry, agent_name, augmented_prompt, model, effort=effort,
                             debate_id=debate_id, debate_dir=debate_dir)
    except AgentError as e:
        if "code -9" in str(e) or "code -15" in str(e):
            log.info("[%s] Killed by stop signal (re-run)", agent_name)
            return None
        log.error("[%s] Agent failed on re-run with research: %s", agent_name, e)
        return f"**[Agent error on re-run: {e}]**"

    return response


def pre_research_claim(registry: DebateRegistry, state: dict, debate_dir: str) -> list[dict]:
    """Run preliminary research on the claim before Round 1 starts."""
    claim = state["claim"]
    researcher_model = state["settings"]["models"]["researcher"]
    debate_id = state["debate_id"]

    queries = [
        f"{claim} latest news evidence",
        f"{claim} analysis arguments against",
    ]

    results = []
    for query in queries:
        try:
            result = run_researcher(registry, query, claim, "pre_research",
                                    researcher_model, debate_id=debate_id, debate_dir=debate_dir)
        except AgentError as e:
            result = f"[Pre-research failed: {e}]"
            log.warning("Pre-research failed for '%s': %s", query[:50], e)

        entry = {
            "requested_by": "system",
            "query": query,
            "results": result,
            "timestamp": now_iso(),
        }
        results.append(entry)
        append_md(
            os.path.join(debate_dir, "research.md"),
            "Pre-debate Research",
            f"**Query:** {query}\n\n{result}",
        )

    state["pre_research"] = results
    log.info("Pre-research complete: %d queries", len(results))
    return results
