"""Tests for the debate orchestrator: scoring parser, early-stop, refusal detect."""
from __future__ import annotations

import json
import os
import threading

from debater.orchestrator import (
    _aggregate_evaluator_samples,
    _atomic_fact_check_markdown,
    _parse_atomic_claims,
    _parse_verification_questions,
    _try_parse_evaluator_json,
    apply_verdict_guard,
    check_early_stopping,
    classify_claim,
    detect_refusal,
    extract_json_block,
    generate_title_async,
    parse_evaluator_response,
    parse_summarizer_response_json,
    run_analysis_mode,
    run_atomic_fact_check,
    run_debate_cycle,
    run_full_debate,
    run_round_summarizer,
    run_summarizer,
)
from debater.state import create_debate, load_state


class TestParseEvaluatorResponseIndependentScoring:
    """Pin Commit B contract: parser keeps (d, s) independent, no renormalization."""

    def test_no_renormalization_for_weak_weak(self, registry, mock_run_agent):
        # An evaluator legitimately returns 3-4 (both weak); parser must not rescale.
        mock_run_agent.set(lambda agent_name, prompt, **kw:
                           "DEFENDER_SCORE: 3\nSKEPTIC_SCORE: 4\n"
                           "JUSTIFICATION: both weak\nSHIFT: stable")
        out = parse_evaluator_response(registry, "raw evaluator text")
        assert out["defender_score"] == 3
        assert out["skeptic_score"] == 4
        assert out["justification"] == "both weak"
        assert out["shift"] == "stable"

    def test_no_renormalization_for_strong_strong(self, registry, mock_run_agent):
        mock_run_agent.set(lambda agent_name, prompt, **kw:
                           "DEFENDER_SCORE: 8\nSKEPTIC_SCORE: 9\n"
                           "JUSTIFICATION: both strong\nSHIFT: x")
        out = parse_evaluator_response(registry, "raw")
        assert out["defender_score"] == 8
        assert out["skeptic_score"] == 9

    def test_clamps_above_10(self, registry, mock_run_agent):
        mock_run_agent.set(lambda agent_name, prompt, **kw:
                           "DEFENDER_SCORE: 99\nSKEPTIC_SCORE: 50\n"
                           "JUSTIFICATION: x\nSHIFT: y")
        out = parse_evaluator_response(registry, "raw")
        assert 0 <= out["defender_score"] <= 10
        assert 0 <= out["skeptic_score"] <= 10

    def test_falls_back_to_5_5_when_unparseable(self, registry, mock_run_agent):
        mock_run_agent.set(lambda agent_name, prompt, **kw: "complete garbage no labels")
        out = parse_evaluator_response(registry, "raw")
        assert out["defender_score"] == 5
        assert out["skeptic_score"] == 5
        assert out["justification"] == "Evaluation unavailable"

    def test_falls_back_when_run_agent_raises(self, registry, monkeypatch):
        def boom(*a, **kw):
            raise RuntimeError("api dead")
        monkeypatch.setattr("debater.orchestrator.run_agent", boom)
        out = parse_evaluator_response(registry, "raw")
        assert out["defender_score"] == 5
        assert out["skeptic_score"] == 5


class TestCheckEarlyStopping:
    def test_not_triggered_before_threshold(self):
        state = {
            "settings": {"early_stop_threshold": 3},
            "rounds": [
                {"evaluator": {"defender_score": 5, "skeptic_score": 5}},
                {"evaluator": {"defender_score": 5, "skeptic_score": 5}},
            ],
        }
        assert check_early_stopping(state) is False

    def test_triggered_when_threshold_rounds_identical(self):
        state = {
            "settings": {"early_stop_threshold": 3},
            "rounds": [
                {"evaluator": {"defender_score": 7, "skeptic_score": 3}},
                {"evaluator": {"defender_score": 4, "skeptic_score": 6}},
                {"evaluator": {"defender_score": 4, "skeptic_score": 6}},
                {"evaluator": {"defender_score": 4, "skeptic_score": 6}},
            ],
        }
        assert check_early_stopping(state) is True

    def test_not_triggered_when_scores_drift(self):
        state = {
            "settings": {"early_stop_threshold": 3},
            "rounds": [
                {"evaluator": {"defender_score": 4, "skeptic_score": 6}},
                {"evaluator": {"defender_score": 4, "skeptic_score": 6}},
                {"evaluator": {"defender_score": 5, "skeptic_score": 5}},
            ],
        }
        assert check_early_stopping(state) is False

    def test_independent_scoring_compatible(self):
        """With Commit B, scores don't sum to 10. Early stop must still work."""
        state = {
            "settings": {"early_stop_threshold": 3},
            "rounds": [
                {"evaluator": {"defender_score": 3, "skeptic_score": 4}},
                {"evaluator": {"defender_score": 3, "skeptic_score": 4}},
                {"evaluator": {"defender_score": 3, "skeptic_score": 4}},
            ],
        }
        assert check_early_stopping(state) is True

    def test_ignores_rounds_without_evaluator(self):
        state = {
            "settings": {"early_stop_threshold": 3},
            "rounds": [
                {"evaluator": {"defender_score": 4, "skeptic_score": 6}},
                {"evaluator": None},  # incomplete
                {"evaluator": {"defender_score": 4, "skeptic_score": 6}},
            ],
        }
        # Only 2 evaluated rounds, below threshold
        assert check_early_stopping(state) is False


class TestDetectRefusal:
    def test_short_response_not_refusal(self, registry, mock_run_agent):
        assert detect_refusal(registry, "short") is False

    def test_empty_response_not_refusal(self, registry, mock_run_agent):
        assert detect_refusal(registry, "") is False

    def test_yes_means_refusal(self, registry, mock_run_agent):
        mock_run_agent.set(lambda **kw: "YES")
        assert detect_refusal(registry, "I cannot defend this claim because..." * 5) is True

    def test_no_means_not_refusal(self, registry, mock_run_agent):
        mock_run_agent.set(lambda **kw: "NO")
        assert detect_refusal(registry, "Here is my argument for the claim..." * 5) is False

    def test_exception_is_not_refusal(self, registry, monkeypatch):
        def boom(*a, **kw):
            raise RuntimeError("broken")
        monkeypatch.setattr("debater.orchestrator.run_agent", boom)
        # Errors should default to False so a legitimate response isn't dropped
        assert detect_refusal(registry, "actual content " * 10) is False


def _agent_responder(canned: dict[str, str]):
    """Return a responder function for mock_run_agent that dispatches by agent_name.

    `canned` keys may be exact agent names (e.g. "defender") or fall through to "_default".
    """
    def responder(agent_name, prompt, **kw):
        return canned.get(agent_name, canned.get("_default", "ok response from " + agent_name))
    return responder


class TestGenerateTitleAsync:
    def test_writes_title_to_state(self, registry, tmp_debates_dir, mock_run_agent):
        debate_id = create_debate(tmp_debates_dir, "trump will win the election", {})
        debate_dir = os.path.join(tmp_debates_dir, debate_id)
        mock_run_agent.set(_agent_responder({"titler": "  \"Election Prediction\"  "}))
        generate_title_async(registry, debate_dir)
        assert load_state(debate_dir)["title"] == "Election Prediction"

    def test_noop_when_title_already_set(self, registry, tmp_debates_dir, mock_run_agent):
        debate_id = create_debate(tmp_debates_dir, "x", {})
        debate_dir = os.path.join(tmp_debates_dir, debate_id)
        # Pre-set a title
        state = load_state(debate_dir)
        state["title"] = "preexisting"
        from debater.state import save_state
        save_state(debate_dir, state)

        generate_title_async(registry, debate_dir)
        assert load_state(debate_dir)["title"] == "preexisting"
        # And no LLM call was made
        assert mock_run_agent.calls == []

    def test_exception_swallowed(self, registry, tmp_debates_dir, monkeypatch):
        debate_id = create_debate(tmp_debates_dir, "x", {})
        debate_dir = os.path.join(tmp_debates_dir, debate_id)

        def boom(*a, **kw):
            raise RuntimeError("titler dead")
        monkeypatch.setattr("debater.orchestrator.run_agent", boom)
        # Must not raise — best-effort
        generate_title_async(registry, debate_dir)


class TestRunRoundSummarizer:
    def test_happy_path_returns_trimmed_summary(self, registry, tmp_debates_dir, mock_run_agent):
        debate_id = create_debate(tmp_debates_dir, "x", {})
        debate_dir = os.path.join(tmp_debates_dir, debate_id)
        state = load_state(debate_dir)
        state["rounds"] = [{
            "round": 1,
            "evaluator": {"defender_score": 4, "skeptic_score": 6,
                          "justification": "skeptic stronger", "shift": "first"},
        }]
        mock_run_agent.set(_agent_responder({"round_summarizer": "  The skeptic won round 1.  "}))
        out = run_round_summarizer(registry, state, 1,
                                   state["rounds"][0]["evaluator"], debate_dir)
        assert out == "The skeptic won round 1."

    def test_fallback_to_justification_on_error(self, registry, tmp_debates_dir, monkeypatch):
        debate_id = create_debate(tmp_debates_dir, "x", {})
        debate_dir = os.path.join(tmp_debates_dir, debate_id)
        state = load_state(debate_dir)
        state["rounds"] = [{"round": 1, "evaluator": {
            "defender_score": 4, "skeptic_score": 6,
            "justification": "fallback message", "shift": "x",
        }}]

        def boom(*a, **kw):
            raise RuntimeError("dead")
        monkeypatch.setattr("debater.orchestrator.run_agent", boom)
        out = run_round_summarizer(registry, state, 1,
                                   state["rounds"][0]["evaluator"], debate_dir)
        assert out == "fallback message"


class TestRunSummarizer:
    def test_writes_summary_and_verdict(self, registry, tmp_debates_dir, mock_run_agent):
        debate_id = create_debate(tmp_debates_dir, "x is true", {})
        debate_dir = os.path.join(tmp_debates_dir, debate_id)
        state = load_state(debate_dir)
        state["rounds"] = [{
            "round": 1, "defender": "d", "skeptic": "s",
            "evaluator": {"defender_score": 7, "skeptic_score": 3,
                          "justification": "defender stronger"},
        }]

        mock_run_agent.set(_agent_responder({
            "summarizer": "## Verdict\n**LIKELY TRUE**\nHigh confidence (85%).",
            "verdict_extractor": "LIKELY TRUE",
            "confidence_extractor": "85",
        }))
        run_summarizer(registry, debate_dir, state)

        loaded = load_state(debate_dir)
        assert loaded["verdict"] == "LIKELY TRUE"
        assert loaded["confidence"] == 85
        assert "LIKELY TRUE" in loaded["summary"]
        # summary.md was written
        with open(os.path.join(debate_dir, "summary.md")) as f:
            assert "LIKELY TRUE" in f.read()


def _atomic_fc_responder(canned: dict[str, str]):
    """Helper for tests that exercise the atomic-fact-check pipeline.

    Provides reasonable defaults for `decomposer`, `verifier_questioner`,
    `atomic_fact_checker`, and `researcher` so tests that don't care about
    the FC details can stay terse. Test-specific overrides win.
    """
    defaults = {
        "decomposer": (
            "Extracted these atomic claims:\n\n"
            "```json\n"
            "[{\"claim_id\":\"cl_def_001\",\"side\":\"defender\","
            "\"text\":\"defender atomic claim\",\"key_terms\":[\"x\"]}]\n"
            "```"
        ),
        "verifier_questioner": "```json\n[\"what is x?\"]\n```",
        "atomic_fact_checker": (
            "```json\n"
            "{\"verdict\":\"VERIFIED\",\"confidence\":0.8,"
            "\"evidence_url\":\"https://example.org\","
            "\"source_tier\":\"PRIMARY\",\"rationale\":\"confirmed\"}\n"
            "```"
        ),
        "researcher": "Search result body. https://example.org",
    }
    merged = {**defaults, **canned}
    return _agent_responder(merged)


class TestRunDebateCycle:
    def test_full_cycle_populates_all_agents(self, registry, tmp_debates_dir, mock_run_agent):
        debate_id = create_debate(tmp_debates_dir, "claim", {})
        debate_dir = os.path.join(tmp_debates_dir, debate_id)
        state = load_state(debate_dir)
        state["current_round"] = 1
        from debater.state import save_state
        save_state(debate_dir, state)

        mock_run_agent.set(_atomic_fc_responder({
            "defender": "Defender argues...",
            "skeptic": "Skeptic critiques...",
            "evaluator": "SCORE: defender=4, skeptic=6\n"
                         "JUSTIFICATION: skeptic edges out\nSHIFT: first",
            "eval_parser": "DEFENDER_SCORE: 4\nSKEPTIC_SCORE: 6\n"
                           "JUSTIFICATION: skeptic edges\nSHIFT: first",
            "refusal_detector": "NO",
            "round_summarizer": "Skeptic won by edging out on evidence.",
        }))

        result = run_debate_cycle(registry, debate_dir, state, threading.Event())
        # No early stop on round 1 with default threshold of 3
        assert result is False

        round_data = state["rounds"][-1]
        assert round_data["defender"] == "Defender argues..."
        assert round_data["skeptic"] == "Skeptic critiques..."
        # New: fact_checker markdown is synthesized from the atomic JSON pipeline
        assert "[VERIFIED]" in round_data["fact_checker"]
        assert round_data["fact_checker_atomic"]
        assert round_data["fact_checker_atomic"][0]["verdict"] == "VERIFIED"
        assert round_data["evaluator"]["defender_score"] == 4
        assert round_data["evaluator"]["skeptic_score"] == 6
        assert "Skeptic won" in round_data["round_summary"]

        # Markdown files were written
        for name in ("defender.md", "skeptic.md", "fact_checker.md", "evaluator.md"):
            assert os.path.isfile(os.path.join(debate_dir, name))
        # JSON sidecar was also written for the atomic FC
        assert os.path.isfile(os.path.join(debate_dir, "round_1", "fact_checker.json"))

    def test_independent_scores_persist_through_cycle(
        self, registry, tmp_debates_dir, mock_run_agent,
    ):
        """Commit B regression: evaluator returns (3, 4) and parser must NOT rescale."""
        debate_id = create_debate(tmp_debates_dir, "claim", {})
        debate_dir = os.path.join(tmp_debates_dir, debate_id)
        state = load_state(debate_dir)
        state["current_round"] = 1

        mock_run_agent.set(_agent_responder({
            "defender": "weak defender",
            "skeptic": "weak skeptic",
            "fact_checker": "no claims to check",
            "evaluator": "both weak",
            "eval_parser": "DEFENDER_SCORE: 3\nSKEPTIC_SCORE: 4\n"
                           "JUSTIFICATION: both weak\nSHIFT: first",
            "refusal_detector": "NO",
            "round_summarizer": "Both sides argued poorly.",
        }))

        run_debate_cycle(registry, debate_dir, state, threading.Event())
        eval_out = state["rounds"][-1]["evaluator"]
        # The key assertion: scores are NOT renormalized to sum to 10
        assert eval_out["defender_score"] == 3
        assert eval_out["skeptic_score"] == 4
        assert eval_out["defender_score"] + eval_out["skeptic_score"] == 7

    def test_shutdown_event_aborts_cycle(self, registry, tmp_debates_dir, mock_run_agent):
        debate_id = create_debate(tmp_debates_dir, "x", {})
        debate_dir = os.path.join(tmp_debates_dir, debate_id)
        state = load_state(debate_dir)
        state["current_round"] = 1
        event = threading.Event()
        event.set()  # already stopped
        result = run_debate_cycle(registry, debate_dir, state, event)
        assert result is False
        # The round was appended but never populated
        assert state["rounds"][-1]["defender"] is None


class TestRunFullDebate:
    def test_single_round_max_iterations(self, registry, tmp_debates_dir, mock_run_agent):
        debate_id = create_debate(tmp_debates_dir, "claim", {"max_iterations": 1})
        debate_dir = os.path.join(tmp_debates_dir, debate_id)

        mock_run_agent.set(_agent_responder({
            "defender": "D content",
            "skeptic": "S content",
            "fact_checker": "FC content",
            "evaluator": "ok",
            "eval_parser": "DEFENDER_SCORE: 5\nSKEPTIC_SCORE: 5\n"
                           "JUSTIFICATION: tied\nSHIFT: first",
            "refusal_detector": "NO",
            "round_summarizer": "Round one tied.",
            "researcher": "pre-research result",
            "summarizer": "## Verdict\n**INCONCLUSIVE**\n",
            "verdict_extractor": "INCONCLUSIVE",
            "confidence_extractor": "50",
        }))

        run_full_debate(registry, debate_dir, threading.Event())

        loaded = load_state(debate_dir)
        assert loaded["status"] == "complete"
        assert loaded["verdict"] == "INCONCLUSIVE"
        assert loaded["confidence"] == 50
        assert len(loaded["rounds"]) == 1
        assert loaded["rounds"][0]["evaluator"]["defender_score"] == 5

    def test_handles_load_state_failure(self, registry, tmp_path):
        # Pass a debate dir that doesn't exist → load_state returns None → early return
        run_full_debate(registry, str(tmp_path / "missing"), threading.Event())
        # No exception means pass


class TestRunAnalysisMode:
    def test_writes_verdict_and_analysis(self, registry, tmp_debates_dir, mock_run_agent):
        debate_id = create_debate(tmp_debates_dir, "controversial claim", {})
        debate_dir = os.path.join(tmp_debates_dir, debate_id)
        state = load_state(debate_dir)

        mock_run_agent.set(_agent_responder({
            "analyst": "## Verdict\n**REFUTED**\nThorough analysis here.",
            "verdict_extractor": "REFUTED",
            "confidence_extractor": "95",
        }))

        run_analysis_mode(registry, debate_dir, state)

        loaded = load_state(debate_dir)
        assert loaded["status"] == "complete"
        assert loaded["verdict"] == "REFUTED"
        assert loaded["confidence"] == 95
        assert "Thorough analysis" in loaded["analysis"]
        assert loaded["analysis_entry_reason"] == "refusal"
        # analysis.md + summary.md both written
        assert os.path.isfile(os.path.join(debate_dir, "analysis.md"))
        assert os.path.isfile(os.path.join(debate_dir, "summary.md"))

    def test_entry_reason_classifier_changes_summary_note(
        self, registry, tmp_debates_dir, mock_run_agent,
    ):
        debate_id = create_debate(tmp_debates_dir, "vaccines cause autism", {})
        debate_dir = os.path.join(tmp_debates_dir, debate_id)
        state = load_state(debate_dir)

        mock_run_agent.set(_agent_responder({
            "analyst": "## Verdict\n**REFUTED**\nAnalysis with citations.",
            "verdict_extractor": "REFUTED",
            "confidence_extractor": "95",
        }))

        run_analysis_mode(registry, debate_dir, state, entry_reason="classifier")

        loaded = load_state(debate_dir)
        assert loaded["analysis_entry_reason"] == "classifier"
        with open(os.path.join(debate_dir, "summary.md")) as f:
            note = f.read()
        assert "HARM-INDUCING FALSEHOOD" in note
        assert "declined to engage" not in note

    def test_analyst_exception_records_failure_in_analysis(
        self, registry, tmp_debates_dir, monkeypatch,
    ):
        debate_id = create_debate(tmp_debates_dir, "x", {})
        debate_dir = os.path.join(tmp_debates_dir, debate_id)
        state = load_state(debate_dir)

        def boom(*args, **kwargs):
            raise RuntimeError("analyst is down")
        monkeypatch.setattr("debater.orchestrator.run_agent", boom)
        monkeypatch.setattr("debater.research.run_agent", boom)

        run_analysis_mode(registry, debate_dir, state)

        loaded = load_state(debate_dir)
        # Even on inner failure, we still reach "complete" with an error string in analysis.
        assert loaded["status"] == "complete"
        assert "Analysis failed" in (loaded.get("analysis") or "")


class TestExtractJsonBlock:
    def test_valid_json_block(self):
        text = 'preamble\n\n```json\n{"a": 1}\n```\n\npostamble'
        assert extract_json_block(text) == {"a": 1}

    def test_last_json_block_wins(self):
        text = '```json\n{"a": 1}\n```\nlater\n```json\n{"a": 2}\n```'
        assert extract_json_block(text) == {"a": 2}

    def test_malformed_json_returns_none(self):
        text = '```json\n{not valid}\n```'
        assert extract_json_block(text) is None

    def test_missing_fence_returns_none(self):
        assert extract_json_block("just prose, no fence") is None

    def test_empty_input_returns_none(self):
        assert extract_json_block("") is None
        assert extract_json_block(None) is None  # type: ignore[arg-type]

    def test_falls_back_to_bare_fence(self):
        text = 'output:\n```\n[1, 2, 3]\n```'
        assert extract_json_block(text) == [1, 2, 3]

    def test_array_payload(self):
        text = '```json\n[{"x": 1}, {"x": 2}]\n```'
        assert extract_json_block(text) == [{"x": 1}, {"x": 2}]


class TestTryParseEvaluatorJson:
    """Phase 3 fast-path: try to read structured eval JSON straight from output."""

    def test_valid_json_returns_canonical_dict(self):
        text = (
            "REASONING: skeptic stronger\n"
            "SCORE: defender=3, skeptic=8\n\n"
            "```json\n{"
            "\"defender\": {\"score\":3, \"confidence\":0.8, \"rationale\":\"weak D\"},"
            "\"skeptic\":  {\"score\":8, \"confidence\":0.9, \"rationale\":\"strong S\"},"
            "\"evidence_used\":[\"cl_def_001\", \"cl_skp_002\"]"
            "}\n```"
        )
        out = _try_parse_evaluator_json(text)
        assert out["defender_score"] == 3
        assert out["skeptic_score"] == 8
        assert out["defender_confidence"] == 0.8
        assert out["skeptic_confidence"] == 0.9
        assert out["evidence_used"] == ["cl_def_001", "cl_skp_002"]
        assert "weak D" in out["justification"]
        assert "strong S" in out["justification"]

    def test_missing_json_returns_none(self):
        assert _try_parse_evaluator_json("just markdown, no fence") is None

    def test_malformed_json_returns_none(self):
        text = "```json\n{not valid}\n```"
        assert _try_parse_evaluator_json(text) is None

    def test_missing_side_dict_returns_none(self):
        text = '```json\n{"defender": {"score": 5}}\n```'
        assert _try_parse_evaluator_json(text) is None

    def test_clamps_scores_to_0_10(self):
        text = '```json\n{"defender":{"score":99},"skeptic":{"score":-3}}\n```'
        out = _try_parse_evaluator_json(text)
        assert out["defender_score"] == 10
        assert out["skeptic_score"] == 0

    def test_confidence_defaults_when_missing(self):
        text = '```json\n{"defender":{"score":5},"skeptic":{"score":5}}\n```'
        out = _try_parse_evaluator_json(text)
        assert 0.0 <= out["defender_confidence"] <= 1.0


class TestParseEvaluatorResponseJsonFirst:
    """End-to-end: parse_evaluator_response prefers JSON, falls back to LLM extractor."""

    def test_json_path_wins_no_eval_parser_call(self, registry, mock_run_agent):
        text = (
            'SCORE: defender=4, skeptic=6\n'
            '```json\n{"defender":{"score":4,"confidence":0.8,"rationale":"d"},'
            '"skeptic":{"score":6,"confidence":0.9,"rationale":"s"},'
            '"evidence_used":["cl_def_001"]}\n```'
        )
        out = parse_evaluator_response(registry, text)
        assert out["defender_score"] == 4
        assert out["skeptic_score"] == 6
        assert out["defender_confidence"] == 0.8
        assert out["evidence_used"] == ["cl_def_001"]
        # eval_parser LLM was NOT invoked (JSON fast path)
        assert all(c["agent_name"] != "eval_parser" for c in mock_run_agent.calls)

    def test_falls_back_to_llm_extractor_when_no_json(self, registry, mock_run_agent):
        mock_run_agent.set(lambda agent_name, prompt, **kw:
                           "DEFENDER_SCORE: 5\nSKEPTIC_SCORE: 5\n"
                           "JUSTIFICATION: tied\nSHIFT: x")
        out = parse_evaluator_response(registry, "old-style markdown only")
        assert out["defender_score"] == 5
        # Fallback path uses default 0.7 confidence
        assert out["defender_confidence"] == 0.7


class TestParseSummarizerResponseJson:
    def test_valid_returns_structured(self):
        text = (
            "### Verdict\n**LIKELY FALSE**\nblah blah\n\n"
            "```json\n{"
            '"verdict": "LIKELY FALSE", "verdict_confidence": 0.82,'
            '"key_claims_cited": ["cl_def_001"], "rationale": "fact-check failures"'
            "}\n```"
        )
        out = parse_summarizer_response_json(text)
        assert out["verdict"] == "LIKELY FALSE"
        assert out["verdict_confidence"] == 0.82
        assert out["key_claims_cited"] == ["cl_def_001"]

    def test_unknown_verdict_returns_none(self):
        text = '```json\n{"verdict":"MAYBE","verdict_confidence":0.5}\n```'
        assert parse_summarizer_response_json(text) is None

    def test_missing_block_returns_none(self):
        assert parse_summarizer_response_json("just prose") is None

    def test_malformed_json_returns_none(self):
        assert parse_summarizer_response_json("```json\n{bad}\n```") is None

    def test_confidence_clamped_to_unit_interval(self):
        text = '```json\n{"verdict":"INCONCLUSIVE","verdict_confidence":1.7}\n```'
        out = parse_summarizer_response_json(text)
        assert out["verdict_confidence"] == 1.0


class TestRunSummarizerJsonFastPath:
    def test_uses_json_verdict_when_present(
        self, registry, tmp_debates_dir, mock_run_agent,
    ):
        debate_id = create_debate(tmp_debates_dir, "x", {})
        debate_dir = os.path.join(tmp_debates_dir, debate_id)
        state = load_state(debate_dir)
        state["rounds"] = [{"round": 1, "defender": "d", "skeptic": "s"}]

        mock_run_agent.set(_agent_responder({
            "summarizer": (
                "### Verdict\n**REFUTED**\nbecause X\n\n"
                "```json\n"
                "{\"verdict\":\"REFUTED\",\"verdict_confidence\":0.91,"
                "\"key_claims_cited\":[\"cl_skp_001\"],\"rationale\":\"r\"}\n"
                "```"
            ),
            # LLM extractors should NOT be called when JSON is present
            "verdict_extractor": "(SHOULD NOT BE CALLED)",
            "confidence_extractor": "(SHOULD NOT BE CALLED)",
        }))

        run_summarizer(registry, debate_dir, state)
        loaded = load_state(debate_dir)
        assert loaded["verdict"] == "REFUTED"
        assert loaded["confidence"] == 91
        assert loaded["verdict_confidence"] == 0.91
        assert loaded["key_claims_cited"] == ["cl_skp_001"]
        # Confirm the LLM extractors were not invoked
        names = [c["agent_name"] for c in mock_run_agent.calls]
        assert "verdict_extractor" not in names
        assert "confidence_extractor" not in names

    def test_falls_back_to_llm_extractors_when_no_json(
        self, registry, tmp_debates_dir, mock_run_agent,
    ):
        debate_id = create_debate(tmp_debates_dir, "x", {})
        debate_dir = os.path.join(tmp_debates_dir, debate_id)
        state = load_state(debate_dir)
        state["rounds"] = [{"round": 1, "defender": "d", "skeptic": "s"}]

        mock_run_agent.set(_agent_responder({
            "summarizer": "### Verdict\n**LIKELY TRUE**\n no JSON block here",
            "verdict_extractor": "LIKELY TRUE",
            "confidence_extractor": "82",
        }))
        run_summarizer(registry, debate_dir, state)
        loaded = load_state(debate_dir)
        assert loaded["verdict"] == "LIKELY TRUE"
        assert loaded["confidence"] == 82


class TestParseAtomicClaims:
    def test_caps_per_side(self):
        items = [{"claim_id": f"cl_{i}", "side": "defender",
                  "text": f"claim {i}", "key_terms": []}
                 for i in range(10)]
        response = f"```json\n{json.dumps(items)}\n```"
        out = _parse_atomic_claims(response, max_per_side=3)
        assert sum(1 for r in out if r["side"] == "defender") == 3

    def test_skips_unknown_sides(self):
        response = (
            '```json\n['
            '{"claim_id":"a","side":"narrator","text":"x"},'
            '{"claim_id":"b","side":"defender","text":"y"}'
            ']\n```'
        )
        out = _parse_atomic_claims(response, max_per_side=4)
        assert len(out) == 1
        assert out[0]["side"] == "defender"

    def test_skips_blank_text(self):
        response = '```json\n[{"side":"defender","text":""}]\n```'
        assert _parse_atomic_claims(response, max_per_side=4) == []

    def test_assigns_missing_claim_id(self):
        response = '```json\n[{"side":"skeptic","text":"hello"}]\n```'
        out = _parse_atomic_claims(response, max_per_side=4)
        assert out[0]["claim_id"].startswith("cl_skp_")

    def test_unparseable_returns_empty(self):
        assert _parse_atomic_claims("no json here", max_per_side=4) == []


class TestParseVerificationQuestions:
    def test_json_list(self):
        out = _parse_verification_questions('```json\n["q1", "q2"]\n```', 3)
        assert out == ["q1", "q2"]

    def test_caps_to_max(self):
        out = _parse_verification_questions('```json\n["q1","q2","q3","q4"]\n```', 2)
        assert out == ["q1", "q2"]

    def test_bullet_fallback(self):
        text = "Here are questions:\n- what is x?\n- when did y happen?\n- where is z?"
        out = _parse_verification_questions(text, 3)
        assert out == ["what is x?", "when did y happen?", "where is z?"]


class TestRunAtomicFactCheck:
    def test_end_to_end_writes_sidecar_and_markdown(
        self, registry, tmp_debates_dir, mock_run_agent,
    ):
        debate_id = create_debate(tmp_debates_dir, "x", {})
        debate_dir = os.path.join(tmp_debates_dir, debate_id)
        state = load_state(debate_dir)
        round_data = {
            "round": 1,
            "defender": "Defender says something.",
            "skeptic": "Skeptic disputes it.",
            "research": [],
        }

        mock_run_agent.set(_atomic_fc_responder({
            "decomposer": (
                "```json\n["
                "{\"claim_id\":\"cl_def_001\",\"side\":\"defender\","
                "\"text\":\"earth is round\",\"key_terms\":[\"earth\"]},"
                "{\"claim_id\":\"cl_skp_001\",\"side\":\"skeptic\","
                "\"text\":\"earth is flat\",\"key_terms\":[\"earth\"]}"
                "]\n```"
            ),
        }))

        run_atomic_fact_check(registry, state, debate_dir, round_data)

        assert len(round_data["fact_checker_atomic"]) == 2
        assert round_data["fact_checker_atomic"][0]["verdict"] == "VERIFIED"
        assert round_data["fact_checker_atomic"][0]["source_tier"] == "PRIMARY"
        assert "[VERIFIED]" in round_data["fact_checker"]
        # Sidecar JSON written
        sidecar = os.path.join(debate_dir, "round_1", "fact_checker.json")
        assert os.path.isfile(sidecar)
        with open(sidecar) as f:
            saved = json.load(f)
        assert len(saved) == 2

    def test_no_atomic_claims_falls_back_gracefully(
        self, registry, tmp_debates_dir, mock_run_agent,
    ):
        debate_id = create_debate(tmp_debates_dir, "x", {})
        debate_dir = os.path.join(tmp_debates_dir, debate_id)
        state = load_state(debate_dir)
        round_data = {"round": 1, "defender": "d", "skeptic": "s", "research": []}

        mock_run_agent.set(_agent_responder({
            "decomposer": "no JSON here at all, sorry",
        }))

        run_atomic_fact_check(registry, state, debate_dir, round_data)
        assert round_data["fact_checker_atomic"] == []
        assert "could be extracted" in round_data["fact_checker"]

    def test_no_content_skips_all_calls(
        self, registry, tmp_debates_dir, mock_run_agent,
    ):
        debate_id = create_debate(tmp_debates_dir, "x", {})
        debate_dir = os.path.join(tmp_debates_dir, debate_id)
        state = load_state(debate_dir)
        round_data = {"round": 1, "defender": "", "skeptic": "  ", "research": []}

        run_atomic_fact_check(registry, state, debate_dir, round_data)
        assert round_data["fact_checker_atomic"] == []
        assert "No content" in round_data["fact_checker"]
        # No agent calls were issued
        assert not mock_run_agent.calls

    def test_verifier_questioner_prompt_only_carries_atomic_claim(self):
        """Chain-of-Verification: the question generator gets ONLY the atomic claim, not debate text."""
        from debater.prompts import build_verification_question_prompt
        atomic = "Wakefield's 1998 study was retracted by The Lancet in 2010."
        prompt = build_verification_question_prompt(atomic, 2)
        # The atomic claim is present.
        assert atomic in prompt
        # The prompt must NOT carry debate-round text that the caller might pass
        # alongside the claim. Sentinel strings from a hypothetical round must
        # not be findable in the prompt.
        debate_sentinels = (
            "Defender says vaccines are unsafe because XYZ.",
            "Skeptic responds that ABC studies refute this.",
        )
        for sentinel in debate_sentinels:
            assert sentinel not in prompt


class TestAtomicFactCheckMarkdown:
    def test_renders_each_claim_one_line(self):
        records = [
            {"claim_id": "cl_def_001", "side": "defender", "text": "earth is round",
             "verdict": "VERIFIED", "confidence": 0.9, "evidence_url": "https://x",
             "source_tier": "PRIMARY", "rationale": ""},
            {"claim_id": "cl_skp_001", "side": "skeptic", "text": "earth is flat",
             "verdict": "FALSE", "confidence": 0.99, "evidence_url": "",
             "source_tier": "PRIMARY", "rationale": "thoroughly disproven"},
        ]
        md = _atomic_fact_check_markdown(records)
        assert "[VERIFIED]" in md and "cl_def_001" in md
        assert "[FALSE]" in md and "cl_skp_001" in md
        assert "thoroughly disproven" in md

    def test_empty_returns_friendly_string(self):
        assert "No atomic claims" in _atomic_fact_check_markdown([])


def _state_with_score_history(score_pairs, claim_class="FACTUAL_TESTABLE",
                              false_def=0, false_skp=0, verdict="INCONCLUSIVE"):
    """Build a synthetic state suitable for apply_verdict_guard exercise."""
    rounds = []
    for i, (d, s) in enumerate(score_pairs, start=1):
        atomic = []
        for j in range(false_def):
            atomic.append({"side": "defender", "verdict": "FALSE",
                           "claim_id": f"cl_def_r{i}_{j}", "text": "x"})
        for j in range(false_skp):
            atomic.append({"side": "skeptic", "verdict": "FALSE",
                           "claim_id": f"cl_skp_r{i}_{j}", "text": "x"})
        rounds.append({
            "round": i,
            "evaluator": {"defender_score": d, "skeptic_score": s,
                          "justification": "x"},
            "fact_checker_atomic": atomic if i == 1 else [],
        })
    return {
        "claim_class": claim_class,
        "verdict": verdict,
        "rounds": rounds,
    }


class TestAggregateEvaluatorSamples:
    """Phase 5 self-consistency aggregator."""

    @staticmethod
    def _sample(d, s, dconf=0.7, sconf=0.7, evidence=None, justification="x"):
        return {
            "defender_score": d, "skeptic_score": s,
            "defender_confidence": dconf, "skeptic_confidence": sconf,
            "evidence_used": list(evidence or []),
            "justification": justification, "shift": "x", "reasoning": "r",
        }

    def test_single_sample_returned_unchanged(self):
        single = self._sample(4, 6)
        out = _aggregate_evaluator_samples([single])
        assert out["defender_score"] == 4
        assert out["skeptic_score"] == 6
        assert out["evaluator_unstable"] is False

    def test_median_of_three(self):
        samples = [
            self._sample(3, 7),
            self._sample(4, 6),
            self._sample(5, 5),
        ]
        out = _aggregate_evaluator_samples(samples)
        assert out["defender_score"] == 4  # median of [3,4,5]
        assert out["skeptic_score"] == 6  # median of [5,6,7]

    def test_high_spread_flags_unstable(self):
        samples = [
            self._sample(2, 8),
            self._sample(5, 5),
            self._sample(9, 2),
        ]
        out = _aggregate_evaluator_samples(samples)
        # spread on defender = 9-2 = 7 > 3
        assert out["evaluator_unstable"] is True

    def test_low_spread_not_unstable(self):
        samples = [
            self._sample(4, 6),
            self._sample(5, 5),
            self._sample(5, 6),
        ]
        out = _aggregate_evaluator_samples(samples)
        assert out["evaluator_unstable"] is False

    def test_evidence_merged_dedup_preserves_order(self):
        samples = [
            self._sample(5, 5, evidence=["cl_def_001", "cl_skp_002"]),
            self._sample(5, 5, evidence=["cl_def_001", "cl_skp_003"]),
        ]
        out = _aggregate_evaluator_samples(samples)
        assert out["evidence_used"] == ["cl_def_001", "cl_skp_002", "cl_skp_003"]

    def test_confidence_is_mean(self):
        samples = [
            self._sample(5, 5, dconf=0.6, sconf=0.9),
            self._sample(5, 5, dconf=0.8, sconf=0.7),
        ]
        out = _aggregate_evaluator_samples(samples)
        assert abs(out["defender_confidence"] - 0.7) < 1e-9
        assert abs(out["skeptic_confidence"] - 0.8) < 1e-9

    def test_representative_justification_picks_closest_to_median(self):
        samples = [
            self._sample(2, 8, justification="extreme low"),
            self._sample(5, 5, justification="middle"),
            self._sample(8, 2, justification="extreme high"),
        ]
        out = _aggregate_evaluator_samples(samples)
        # Median is (5,5); representative is the second sample
        assert out["justification"] == "middle"

    def test_empty_input_returns_safe_default(self):
        out = _aggregate_evaluator_samples([])
        assert out["defender_score"] == 5
        assert out["skeptic_score"] == 5


class TestEvaluatorSelfConsistencyWiring:
    def test_default_runs_three_evaluator_samples(
        self, registry, tmp_debates_dir, mock_run_agent,
    ):
        debate_id = create_debate(tmp_debates_dir, "claim", {})
        debate_dir = os.path.join(tmp_debates_dir, debate_id)
        state = load_state(debate_dir)
        state["current_round"] = 1
        from debater.state import save_state
        save_state(debate_dir, state)

        mock_run_agent.set(_atomic_fc_responder({
            "defender": "D", "skeptic": "S",
            "evaluator": "SCORE: defender=4, skeptic=6\nJUSTIFICATION: x\nSHIFT: x",
            "eval_parser": "DEFENDER_SCORE: 4\nSKEPTIC_SCORE: 6\n"
                           "JUSTIFICATION: x\nSHIFT: x",
            "refusal_detector": "NO",
            "round_summarizer": "tied",
        }))

        run_debate_cycle(registry, debate_dir, state, threading.Event())

        # Three evaluator calls were made (default samples=3)
        eval_calls = [c for c in mock_run_agent.calls if c["agent_name"] == "evaluator"]
        assert len(eval_calls) == 3
        # round_data["evaluator_samples"] has 3 entries
        round_data = state["rounds"][-1]
        assert len(round_data["evaluator_samples"]) == 3
        assert round_data["evaluator"]["defender_score"] == 4
        assert round_data["evaluator"]["skeptic_score"] == 6

    def test_setting_override_changes_sample_count(
        self, registry, tmp_debates_dir, mock_run_agent,
    ):
        debate_id = create_debate(tmp_debates_dir, "claim",
                                   {"evaluator_samples": 1})
        debate_dir = os.path.join(tmp_debates_dir, debate_id)
        state = load_state(debate_dir)
        state["current_round"] = 1
        from debater.state import save_state
        save_state(debate_dir, state)

        mock_run_agent.set(_atomic_fc_responder({
            "defender": "D", "skeptic": "S",
            "evaluator": "SCORE: defender=4, skeptic=6\nJUSTIFICATION: x\nSHIFT: x",
            "eval_parser": "DEFENDER_SCORE: 4\nSKEPTIC_SCORE: 6\n"
                           "JUSTIFICATION: x\nSHIFT: x",
            "refusal_detector": "NO",
            "round_summarizer": "tied",
        }))

        run_debate_cycle(registry, debate_dir, state, threading.Event())
        eval_calls = [c for c in mock_run_agent.calls if c["agent_name"] == "evaluator"]
        assert len(eval_calls) == 1


class TestApplyVerdictGuard:
    """Phase 4: deterministic guard that flips INCONCLUSIVE on strong directional evidence."""

    def test_triggers_skeptic_ahead_flips_to_likely_false(self):
        state = _state_with_score_history(
            score_pairs=[(3, 7), (3, 7), (4, 7), (3, 8)],
            false_def=4, false_skp=0,
        )
        apply_verdict_guard(state)
        assert state["verdict"] == "LIKELY FALSE"
        assert state["verdict_guard_applied"] is True
        assert "losing_false=4" in state["verdict_guard_reason"]

    def test_triggers_defender_ahead_flips_to_likely_true(self):
        state = _state_with_score_history(
            score_pairs=[(8, 3), (7, 3), (8, 4)],
            false_def=0, false_skp=4,
        )
        apply_verdict_guard(state)
        assert state["verdict"] == "LIKELY TRUE"
        assert state["verdict_guard_applied"] is True

    def test_skips_when_gaps_too_small(self):
        state = _state_with_score_history(
            score_pairs=[(5, 6), (5, 6), (5, 6)],
            false_def=4, false_skp=0,
        )
        apply_verdict_guard(state)
        assert state["verdict"] == "INCONCLUSIVE"
        assert state.get("verdict_guard_applied") is not True

    def test_skips_when_insufficient_false_evidence(self):
        state = _state_with_score_history(
            score_pairs=[(3, 7), (3, 7), (3, 7)],
            false_def=1, false_skp=0,
        )
        apply_verdict_guard(state)
        assert state["verdict"] == "INCONCLUSIVE"

    def test_skips_when_winning_side_has_too_many_false(self):
        """If both sides have lots of false claims, the guard does NOT flip."""
        state = _state_with_score_history(
            score_pairs=[(3, 7), (3, 7), (3, 7)],
            false_def=4, false_skp=4,
        )
        apply_verdict_guard(state)
        assert state["verdict"] == "INCONCLUSIVE"

    def test_skips_predictive_future_class(self):
        state = _state_with_score_history(
            score_pairs=[(3, 7), (3, 7), (3, 7)],
            false_def=4, false_skp=0,
            claim_class="PREDICTIVE_FUTURE",
        )
        apply_verdict_guard(state)
        assert state["verdict"] == "INCONCLUSIVE"

    def test_skips_value_normative_class(self):
        state = _state_with_score_history(
            score_pairs=[(3, 7), (3, 7), (3, 7)],
            false_def=4, false_skp=0,
            claim_class="VALUE_NORMATIVE",
        )
        apply_verdict_guard(state)
        assert state["verdict"] == "INCONCLUSIVE"

    def test_skips_when_verdict_already_directional(self):
        state = _state_with_score_history(
            score_pairs=[(3, 7), (3, 7), (3, 7)],
            false_def=4, false_skp=0,
            verdict="REFUTED",
        )
        apply_verdict_guard(state)
        # Verdict is preserved as-is; guard only acts on INCONCLUSIVE.
        assert state["verdict"] == "REFUTED"

    def test_skips_when_no_rounds(self):
        state = {"claim_class": "FACTUAL_TESTABLE", "verdict": "INCONCLUSIVE",
                 "rounds": []}
        apply_verdict_guard(state)
        assert state["verdict"] == "INCONCLUSIVE"


class TestClassifyClaim:
    """Pre-debate claim classifier — routes to one of four classes."""

    def test_returns_each_class_token(self, registry, mock_run_agent):
        for cls in ("FACTUAL_TESTABLE", "PREDICTIVE_FUTURE",
                    "VALUE_NORMATIVE", "HARM_INDUCING_FALSEHOOD"):
            mock_run_agent.set(lambda agent_name, prompt, cls=cls, **kw:
                               f"REASONING: it fits {cls}\nCLASS: {cls}")
            assert classify_claim(registry, "some claim") == cls

    def test_unparseable_defaults_to_factual_testable(self, registry, mock_run_agent):
        mock_run_agent.set(lambda **kw: "i have no idea what this is")
        assert classify_claim(registry, "claim") == "FACTUAL_TESTABLE"

    def test_exception_defaults_to_factual_testable(self, registry, monkeypatch):
        def boom(*a, **kw):
            raise RuntimeError("api dead")
        monkeypatch.setattr("debater.orchestrator.run_agent", boom)
        assert classify_claim(registry, "claim") == "FACTUAL_TESTABLE"

    def test_empty_claim_defaults_to_factual_testable(self, registry, mock_run_agent):
        # Must not call the model on empty input
        assert classify_claim(registry, "") == "FACTUAL_TESTABLE"
        assert classify_claim(registry, "   ") == "FACTUAL_TESTABLE"
        assert mock_run_agent.calls == []

    def test_class_inferred_from_bare_token_in_response(self, registry, mock_run_agent):
        # Model drift: returns the token but skips the "CLASS:" prefix
        mock_run_agent.set(lambda **kw: "this is a PREDICTIVE_FUTURE claim, basically")
        assert classify_claim(registry, "x") == "PREDICTIVE_FUTURE"


class TestRunFullDebateClassifierDispatch:
    """Classifier routes HARM_INDUCING_FALSEHOOD straight to analyst; other classes proceed."""

    def test_harm_inducing_falsehood_skips_debate_runs_analyst(
        self, registry, tmp_debates_dir, mock_run_agent,
    ):
        debate_id = create_debate(tmp_debates_dir, "vaccines cause autism", {"max_iterations": 3})
        debate_dir = os.path.join(tmp_debates_dir, debate_id)

        mock_run_agent.set(_agent_responder({
            "claim_classifier": "REASONING: well-established refuted falsehood\n"
                                "CLASS: HARM_INDUCING_FALSEHOOD",
            "analyst": "## Verdict\n**REFUTED**\nVaccines do not cause autism.",
            "verdict_extractor": "REFUTED",
            "confidence_extractor": "99",
        }))

        run_full_debate(registry, debate_dir, threading.Event())

        loaded = load_state(debate_dir)
        assert loaded["claim_class"] == "HARM_INDUCING_FALSEHOOD"
        assert loaded["status"] == "complete"
        assert loaded["verdict"] == "REFUTED"
        assert loaded["analysis_entry_reason"] == "classifier"
        # Defender / skeptic / fact_checker / evaluator were NEVER called
        called_agents = {c["agent_name"] for c in mock_run_agent.calls}
        for forbidden in ("defender", "skeptic", "fact_checker", "evaluator", "pre_research"):
            assert forbidden not in called_agents, f"{forbidden} should not have been invoked"
        # analyst was called, with HARM-INDUCING framing in its prompt
        analyst_calls = [c for c in mock_run_agent.calls if c["agent_name"] == "analyst"]
        assert analyst_calls
        assert "HARM-INDUCING FALSEHOOD" in analyst_calls[0]["prompt"]
        # analysis.md was written
        assert os.path.isfile(os.path.join(debate_dir, "analysis.md"))

    def test_factual_testable_runs_normal_debate(
        self, registry, tmp_debates_dir, mock_run_agent,
    ):
        debate_id = create_debate(tmp_debates_dir, "boring fact claim", {"max_iterations": 1})
        debate_dir = os.path.join(tmp_debates_dir, debate_id)

        mock_run_agent.set(_agent_responder({
            "claim_classifier": "REASONING: fits testable\nCLASS: FACTUAL_TESTABLE",
            "defender": "D content",
            "skeptic": "S content",
            "fact_checker": "FC content",
            "evaluator": "ok",
            "eval_parser": "DEFENDER_SCORE: 5\nSKEPTIC_SCORE: 5\n"
                           "JUSTIFICATION: tied\nSHIFT: first",
            "refusal_detector": "NO",
            "round_summarizer": "Round one tied.",
            "researcher": "pre-research result",
            "summarizer": "## Verdict\n**INCONCLUSIVE**\n",
            "verdict_extractor": "INCONCLUSIVE",
            "confidence_extractor": "50",
        }))

        run_full_debate(registry, debate_dir, threading.Event())

        loaded = load_state(debate_dir)
        assert loaded["claim_class"] == "FACTUAL_TESTABLE"
        assert loaded["status"] == "complete"
        called_agents = {c["agent_name"] for c in mock_run_agent.calls}
        assert "defender" in called_agents
        assert "skeptic" in called_agents

    def test_classifier_failure_falls_back_to_normal_debate(
        self, registry, tmp_debates_dir, mock_run_agent, monkeypatch,
    ):
        """If the classifier raises mid-call, we must NOT block the debate."""
        debate_id = create_debate(tmp_debates_dir, "claim", {"max_iterations": 1})
        debate_dir = os.path.join(tmp_debates_dir, debate_id)

        original_fake = mock_run_agent.fake
        call_count = {"classifier": 0}

        def selective_boom(*args, **kwargs):
            agent_name = args[1] if len(args) >= 2 else kwargs.get("agent_name", "")
            if agent_name == "claim_classifier":
                call_count["classifier"] += 1
                raise RuntimeError("classifier dead")
            return original_fake(*args, **kwargs)
        monkeypatch.setattr("debater.orchestrator.run_agent", selective_boom)

        mock_run_agent.set(_agent_responder({
            "defender": "D", "skeptic": "S", "fact_checker": "FC",
            "evaluator": "ok",
            "eval_parser": "DEFENDER_SCORE: 5\nSKEPTIC_SCORE: 5\n"
                           "JUSTIFICATION: tied\nSHIFT: x",
            "refusal_detector": "NO", "round_summarizer": "tied",
            "researcher": "ok", "summarizer": "## Verdict\n**INCONCLUSIVE**\n",
            "verdict_extractor": "INCONCLUSIVE", "confidence_extractor": "50",
        }))

        run_full_debate(registry, debate_dir, threading.Event())

        loaded = load_state(debate_dir)
        assert call_count["classifier"] >= 1
        # Defaulted to FACTUAL_TESTABLE → normal debate ran to completion
        assert loaded["claim_class"] == "FACTUAL_TESTABLE"
        assert loaded["status"] == "complete"


class TestRefusalToAnalystDispatch:
    """End-to-end: refused round must produce analysis.md and final status=complete."""

    def test_refusal_dispatches_to_analyst_with_status_complete(
        self, registry, tmp_debates_dir, mock_run_agent,
    ):
        debate_id = create_debate(tmp_debates_dir, "edge claim", {"max_iterations": 3})
        debate_dir = os.path.join(tmp_debates_dir, debate_id)

        mock_run_agent.set(_agent_responder({
            "claim_classifier": "REASONING: testable\nCLASS: FACTUAL_TESTABLE",
            # Defender refuses even after retry.
            "defender": "I cannot in good conscience defend this. " * 5,
            "refusal_detector": "YES",
            # Analyst rescues the debate.
            "analyst": "## Verdict\n**REFUTED**\nFinal analyst verdict.",
            "verdict_extractor": "REFUTED",
            "confidence_extractor": "90",
            "researcher": "pre-research result",
        }))

        run_full_debate(registry, debate_dir, threading.Event())

        loaded = load_state(debate_dir)
        # Regression: status must be "complete", NOT "stopped" or "refused".
        assert loaded["status"] == "complete", (
            f"Expected status=complete after refusal→analyst; got {loaded['status']!r}. "
            "This is the regression for the orchestrator status-reset bug."
        )
        assert loaded["verdict"] == "REFUTED"
        assert loaded["analysis_entry_reason"] == "refusal"
        assert os.path.isfile(os.path.join(debate_dir, "analysis.md"))

    def test_analyst_exception_during_refusal_sets_analysis_failed(
        self, registry, tmp_debates_dir, mock_run_agent, monkeypatch,
    ):
        """If analyst itself raises a real exception (not caught internally),
        the outer dispatcher must record status=analysis_failed, not silently stop."""
        debate_id = create_debate(tmp_debates_dir, "edge claim", {"max_iterations": 3})
        debate_dir = os.path.join(tmp_debates_dir, debate_id)

        # Force the analyst-mode entrypoint itself to raise. We can't easily make
        # run_agent_with_research raise (it catches AgentError), so we patch
        # run_analysis_mode directly.
        def boom(*args, **kwargs):
            raise RuntimeError("disk full")
        monkeypatch.setattr("debater.orchestrator.run_analysis_mode", boom)

        mock_run_agent.set(_agent_responder({
            "claim_classifier": "REASONING: testable\nCLASS: FACTUAL_TESTABLE",
            "defender": "I cannot defend this. " * 5,
            "refusal_detector": "YES",
            "researcher": "pre-research result",
        }))

        run_full_debate(registry, debate_dir, threading.Event())

        loaded = load_state(debate_dir)
        assert loaded["status"] == "analysis_failed"
