"""Tests for prompt templates and context assembly."""
from __future__ import annotations

from debater.prompts import (
    _build_prior_block,
    build_agent_context,
    build_analyst_prompt,
    build_claim_classifier_prompt,
    build_evaluator_prompt,
    build_researcher_prompt,
    build_summarizer_prompt,
    tail_sections,
)


def _make_state(rounds=None, **extra):
    """Helper to assemble a state dict suitable for context-building."""
    base = {
        "claim": "the sky is blue",
        "rounds": rounds or [],
        "user_messages": [],
        "last_seen_user_msg_idx": 0,
        "current_round": (rounds[-1]["round"] if rounds else 1),
        "settings": {
            "max_iterations": 5,
            "models": {"defender": "m", "skeptic": "m", "evaluator": "m"},
            "efforts": {},
        },
    }
    base.update(extra)
    return base


class TestTailSections:
    def test_returns_full_when_few_sections(self):
        content = "header\n## a\nA body\n## b\nB body\n"
        assert tail_sections(content, 5) == content

    def test_returns_header_plus_last_n(self):
        content = "header\n## a\nA\n## b\nB\n## c\nC\n## d\nD\n"
        out = tail_sections(content, 2)
        assert out.startswith("header\n")
        assert "## c" in out
        assert "## d" in out
        assert "## a" not in out
        assert "## b" not in out

    def test_empty_input(self):
        assert tail_sections("", 5) == ""

    def test_n_zero(self):
        content = "anything"
        assert tail_sections(content, 0) == content


class TestPriorBlock:
    def test_empty_returns_empty_string(self):
        assert _build_prior_block({}) == ""

    def test_modern_list_format(self):
        block = _build_prior_block({"prior_debates": [{
            "claim": "X", "verdict": "REFUTED", "summary": "blah",
        }]})
        assert "REFUTED" in block
        assert "X" in block
        assert "Prior Debate 1" in block

    def test_legacy_single_fields(self):
        block = _build_prior_block({
            "prior_debate_summary": "stuff",
            "prior_debate_verdict": "LIKELY TRUE",
            "prior_debate_claim": "Y",
        })
        assert "LIKELY TRUE" in block
        assert "Y" in block

    def test_truncates_long_summaries(self):
        block = _build_prior_block({"prior_debates": [{
            "claim": "X", "verdict": "REFUTED", "summary": "z" * 10000,
        }]})
        # 4000 char cap on summary
        assert len(block) < 5000


class TestBuildAgentContext:
    def test_no_rounds_yet(self):
        ctx = build_agent_context(_make_state(), "defender")
        assert 'CLAIM: "the sky is blue"' in ctx
        assert "(No scores yet)" in ctx
        # No opponent block, no fact-check block, no research block
        assert "OPPONENT'S LATEST ARGUMENT" not in ctx
        assert "FACT-CHECK FINDINGS" not in ctx
        assert "AVAILABLE RESEARCH" not in ctx

    def test_includes_own_history_and_opponent_last(self):
        rounds = [
            {"round": 1, "defender": "D1 says...", "skeptic": "S1 says...",
             "evaluator": {"defender_score": 4, "skeptic_score": 6,
                           "justification": "skeptic better"}},
        ]
        ctx = build_agent_context(_make_state(rounds=rounds), "defender")
        assert "YOUR PREVIOUS ARGUMENTS" in ctx
        assert "D1 says..." in ctx
        assert "OPPONENT'S LATEST ARGUMENT (Round 1)" in ctx
        assert "S1 says..." in ctx
        assert "Defender 4 - Skeptic 6" in ctx

    def test_fact_check_pulled_from_latest_round(self):
        rounds = [
            {"round": 1, "defender": "D1", "skeptic": "S1",
             "fact_checker": "First-round verdict",
             "evaluator": {"defender_score": 5, "skeptic_score": 5,
                           "justification": "tied"}},
        ]
        ctx = build_agent_context(_make_state(rounds=rounds), "defender")
        assert "FACT-CHECK FINDINGS (previous round)" in ctx
        assert "First-round verdict" in ctx

    def test_condensation_kicks_in_at_threshold(self):
        # Three+ rounds where own_chars > 15000 triggers the condense branch
        big_text = "x" * 6000
        rounds = []
        for i in range(1, 5):
            rounds.append({
                "round": i,
                "defender": big_text,
                "skeptic": "S",
                "evaluator": {"defender_score": 5, "skeptic_score": 5,
                              "justification": "x"},
            })
        ctx = build_agent_context(_make_state(rounds=rounds), "defender")
        assert "[... condensed ...]" in ctx
        # Last two rounds shown in full
        assert "Round 3 (full)" in ctx
        assert "Round 4 (full)" in ctx

    def test_user_interjections_surface_when_unseen(self):
        state = _make_state(
            user_messages=[
                {"message": "wait, what about X?", "during_round": 1, "during_agent": "defender"}
            ],
            last_seen_user_msg_idx=0,
        )
        ctx = build_agent_context(state, "skeptic")
        assert "USER INTERJECTIONS" in ctx
        assert "wait, what about X?" in ctx

    def test_user_interjections_hidden_when_already_seen(self):
        state = _make_state(
            user_messages=[
                {"message": "old", "during_round": 1, "during_agent": "defender"}
            ],
            last_seen_user_msg_idx=1,
        )
        ctx = build_agent_context(state, "skeptic")
        assert "USER INTERJECTIONS" not in ctx

    def test_research_block_includes_pre_research(self):
        state = _make_state()
        state["pre_research"] = [
            {"requested_by": "system", "query": "q?", "results": "result body"}
        ]
        ctx = build_agent_context(state, "defender")
        assert "AVAILABLE RESEARCH" in ctx
        assert "result body" in ctx


class TestEvaluatorPromptIndependentScoring:
    """Pin the Commit B contract: no sum-to-10 constraint, rubric present."""

    def _state_with_round(self):
        rounds = [{
            "round": 1, "defender": "D1", "skeptic": "S1",
            "fact_checker": "checked",
        }]
        return _make_state(rounds=rounds, current_round=1)

    def test_no_sum_to_ten_constraint(self):
        prompt = build_evaluator_prompt(self._state_with_round(), 1)
        # The old prompt mandated "MUST sum to exactly 10"; the new one MUST not.
        assert "MUST sum to exactly 10" not in prompt
        assert "sum to exactly 10" not in prompt

    def test_says_scores_are_independent(self):
        prompt = build_evaluator_prompt(self._state_with_round(), 1)
        # New rubric language
        assert "INDEPENDENT" in prompt
        assert "do NOT need to sum" in prompt or "do not need to sum" in prompt.lower()

    def test_rubric_bands_present(self):
        prompt = build_evaluator_prompt(self._state_with_round(), 1)
        # All six bands appear
        for band in ("9–10", "7–8", "5–6", "3–4", "1–2"):
            assert band in prompt, f"missing rubric band {band}"

    def test_no_compress_to_5_5_clause(self):
        prompt = build_evaluator_prompt(self._state_with_round(), 1)
        # The old "Do NOT compress scores toward 5-5" clause is gone
        assert "compress scores toward 5-5" not in prompt


class TestBuildClaimClassifierPrompt:
    def test_lists_all_four_classes(self):
        prompt = build_claim_classifier_prompt("vaccines cause autism")
        for cls in ("FACTUAL_TESTABLE", "PREDICTIVE_FUTURE",
                    "VALUE_NORMATIVE", "HARM_INDUCING_FALSEHOOD"):
            assert cls in prompt

    def test_embeds_the_claim(self):
        prompt = build_claim_classifier_prompt("the S&P will drop")
        assert "the S&P will drop" in prompt

    def test_asks_for_class_line_format(self):
        prompt = build_claim_classifier_prompt("x")
        # Parser scans for "CLASS:" lines; ensure prompt instructs that
        assert "CLASS:" in prompt


class TestBuildAnalystPrompt:
    def test_refusal_opener_default(self):
        prompt = build_analyst_prompt("some claim")
        assert "declined to engage" in prompt
        assert "HARM-INDUCING FALSEHOOD" not in prompt

    def test_classifier_opener_changes_framing(self):
        prompt = build_analyst_prompt("vaccines cause autism", entry_reason="classifier")
        assert "HARM-INDUCING FALSEHOOD" in prompt
        assert "declined to engage" not in prompt

    def test_claim_embedded_in_both_openers(self):
        for reason in ("refusal", "classifier"):
            prompt = build_analyst_prompt("my specific claim XYZ", entry_reason=reason)
            assert "my specific claim XYZ" in prompt

    def test_emits_structured_verdict_json_spec(self):
        prompt = build_analyst_prompt("x")
        assert "```json" in prompt
        assert "verdict_confidence" in prompt


class TestPhase3StructuredOutput:
    """Phase 3: evaluator and summarizer prompts both request a fenced JSON tail."""

    def _state_with_round(self):
        rounds = [{"round": 1, "defender": "D", "skeptic": "S", "fact_checker": ""}]
        return {
            "claim": "x", "rounds": rounds, "user_messages": [],
            "last_seen_user_msg_idx": 0, "current_round": 1,
            "settings": {"max_iterations": 5,
                         "models": {"defender": "m", "skeptic": "m", "evaluator": "m",
                                    "summarizer": "m"},
                         "efforts": {}},
        }

    def test_evaluator_requests_fenced_json(self):
        prompt = build_evaluator_prompt(self._state_with_round(), 1)
        assert "```json" in prompt
        # JSON schema landmarks
        for term in ("confidence", "evidence_used", "rationale"):
            assert term in prompt

    def test_summarizer_requests_fenced_json(self):
        prompt = build_summarizer_prompt(self._state_with_round())
        assert "```json" in prompt
        for term in ("verdict_confidence", "key_claims_cited", "rationale"):
            assert term in prompt


class TestBuildResearcherPromptSourceTier:
    """Phase 5: researcher must request a source-tier tag on each finding."""

    def test_lists_all_four_tiers(self):
        prompt = build_researcher_prompt("q", "c", "defender")
        for tier in ("PRIMARY", "SECONDARY", "TERTIARY", "LOW"):
            assert tier in prompt

    def test_instructs_tier_tag_format(self):
        prompt = build_researcher_prompt("q", "c", "defender")
        assert "[TIER:" in prompt

    def test_embeds_query_and_claim(self):
        prompt = build_researcher_prompt("my-q", "my-claim", "skeptic")
        assert "my-q" in prompt
        assert "my-claim" in prompt
