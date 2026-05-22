"""Tests for research-request extraction and the run_agent_with_research loop."""
from __future__ import annotations

import os

from debater.research import extract_research_requests, run_agent_with_research
from debater.state import create_debate


class TestExtractResearchRequests:
    def test_empty(self):
        assert extract_research_requests("just text") == []

    def test_single_tag(self):
        out = extract_research_requests("foo <research_request>what is X?</research_request> bar")
        assert out == ["what is X?"]

    def test_multiple_tags(self):
        text = (
            "I'd like to verify these.\n"
            "<research_request>claim about A</research_request>\n"
            "Then also:\n"
            "<research_request>claim about B</research_request>\n"
        )
        assert extract_research_requests(text) == ["claim about A", "claim about B"]

    def test_multiline_tag_body(self):
        text = "<research_request>line one\nline two</research_request>"
        out = extract_research_requests(text)
        assert out == ["line one\nline two"]

    def test_malformed_unclosed_tag_ignored(self):
        # Without a closing tag the regex shouldn't match
        assert extract_research_requests("<research_request>oops") == []


class TestRunAgentWithResearch:
    def test_returns_response_when_no_research_requested(self, registry, tmp_debates_dir, mock_run_agent):
        debate_id = create_debate(tmp_debates_dir, "claim", {})
        debate_dir = os.path.join(tmp_debates_dir, debate_id)

        mock_run_agent.set(lambda agent_name, prompt, **kw: f"plain response from {agent_name}")

        state = {"debate_id": debate_id, "claim": "claim",
                 "settings": {"models": {"researcher": "m"}}}
        round_data = {"round": 1, "research": []}

        out = run_agent_with_research(
            registry, "defender", "prompt", "model", None, state, debate_dir, round_data
        )
        assert out == "plain response from defender"
        # One run_agent call: just the initial agent
        assert len(mock_run_agent.calls) == 1
        assert mock_run_agent.calls[0]["agent_name"] == "defender"

    def test_research_loop_runs_searches_and_reruns_agent(
        self, registry, tmp_debates_dir, mock_run_agent,
    ):
        debate_id = create_debate(tmp_debates_dir, "claim", {})
        debate_dir = os.path.join(tmp_debates_dir, debate_id)

        # First defender call requests one research; subsequent calls return plain text.
        call_count = {"n": 0}

        def responder(agent_name, prompt, **kw):
            call_count["n"] += 1
            if agent_name == "defender" and call_count["n"] == 1:
                return "<research_request>what is X?</research_request>"
            if agent_name == "researcher":
                return "X is a thing per [example.com]"
            # Re-run after research:
            return "final defender answer"

        mock_run_agent.set(responder)

        state = {"debate_id": debate_id, "claim": "claim",
                 "settings": {"models": {"researcher": "m"}}}
        round_data = {"round": 1, "research": []}

        out = run_agent_with_research(
            registry, "defender", "P", "M", None, state, debate_dir, round_data,
        )
        assert out == "final defender answer"

        # Three calls in sequence: defender (extract), researcher (answer), defender (rerun).
        seq = [c["agent_name"] for c in mock_run_agent.calls]
        assert seq == ["defender", "researcher", "defender"]

        # The research entry was recorded on round_data.
        assert len(round_data["research"]) == 1
        assert round_data["research"][0]["query"] == "what is X?"
        assert "X is a thing" in round_data["research"][0]["results"]

        # The augmented re-run prompt includes the research results.
        captured = " ".join(str(v) for v in mock_run_agent.calls[2].values()
                            if isinstance(v, (str, dict)))
        assert "RESEARCH RESULTS" in captured
