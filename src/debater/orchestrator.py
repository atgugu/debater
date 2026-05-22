"""Debate orchestration: per-round cycle, summarizer, analyst, and the full debate driver."""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import Any

from debater.claude_runner import (
    debate_log,
    extract_confidence_with_llm,
    extract_verdict_with_llm,
    run_agent,
)
from debater.config import CLAIM_CLASSES, AgentError
from debater.prompts import (
    build_analyst_prompt,
    build_atomic_fact_check_prompt,
    build_claim_classifier_prompt,
    build_decomposer_prompt,
    build_defender_prompt,
    build_evaluator_prompt,
    build_skeptic_prompt,
    build_summarizer_prompt,
    build_verification_question_prompt,
)
from debater.registry import DebateRegistry
from debater.research import pre_research_claim, run_agent_with_research, run_researcher
from debater.state import append_md, load_state, now_iso, save_state

_JSON_FENCE_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_ANY_FENCE_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
_ATOMIC_VERDICTS = ("VERIFIED", "LIKELY", "UNVERIFIED", "DISPUTED", "FALSE")
_SOURCE_TIERS = ("PRIMARY", "SECONDARY", "TERTIARY", "LOW")


def extract_json_block(text: str) -> Any | None:
    """Find the LAST ```json fenced block in `text` and return parsed JSON.

    Falls back to the last bare ``` fence if no ```json fence is found.
    Returns None if no fence is found or if parsing fails.
    """
    if not text:
        return None
    matches = _JSON_FENCE_RE.findall(text)
    if not matches:
        matches = _ANY_FENCE_RE.findall(text)
    if not matches:
        return None
    try:
        return json.loads(matches[-1].strip())
    except json.JSONDecodeError:
        return None

log = logging.getLogger("debater")


def _try_parse_evaluator_json(text: str) -> dict | None:
    """Phase 3 fast-path parser: pull the structured JSON block from evaluator output.

    Returns the canonical evaluation dict (matching the regex parser's shape, plus
    `defender_confidence`, `skeptic_confidence`, and `evidence_used`) or None if
    the JSON is missing / malformed / doesn't carry both side dicts.
    """
    parsed = extract_json_block(text)
    if not isinstance(parsed, dict):
        return None
    d = parsed.get("defender")
    s = parsed.get("skeptic")
    if not isinstance(d, dict) or not isinstance(s, dict):
        return None
    try:
        d_score = max(0, min(10, int(d.get("score", 5))))
        s_score = max(0, min(10, int(s.get("score", 5))))
    except (TypeError, ValueError):
        return None

    def _conf(val, default=0.7):
        try:
            return max(0.0, min(1.0, float(val)))
        except (TypeError, ValueError):
            return default

    d_conf = _conf(d.get("confidence"))
    s_conf = _conf(s.get("confidence"))
    evidence = [str(e).strip() for e in (parsed.get("evidence_used") or [])
                if str(e).strip()][:32]
    d_rationale = str(d.get("rationale") or "").strip()
    s_rationale = str(s.get("rationale") or "").strip()
    if d_rationale and s_rationale:
        justification = f"D: {d_rationale} | S: {s_rationale}"
    else:
        justification = d_rationale or s_rationale or "see narrative"
    return {
        "defender_score": d_score,
        "skeptic_score": s_score,
        "defender_confidence": d_conf,
        "skeptic_confidence": s_conf,
        "evidence_used": evidence,
        "justification": justification[:500],
        "shift": "see narrative",
        "reasoning": text[:500] if text else None,
    }


def parse_evaluator_response(registry: DebateRegistry, text: str,
                             debate_id: str | None = None,
                             debate_dir: str | None = None) -> dict:
    """Extract a structured evaluation dict from the evaluator's response.

    Strategy: try the Phase 3 JSON-in-markdown fast path first; if that's missing
    or malformed, fall back to the legacy LLM-based regex parser. Backward-compat
    with debates that ran before Phase 3 is preserved.
    """
    fast = _try_parse_evaluator_json(text)
    if fast is not None:
        return fast

    log.debug("Evaluator JSON block missing/invalid — falling back to LLM extractor")
    prompt = (
        "Extract the evaluation from the text below. Return EXACTLY 4 lines in this format:\n"
        "DEFENDER_SCORE: N\n"
        "SKEPTIC_SCORE: M\n"
        "JUSTIFICATION: one-line explanation of the score\n"
        "SHIFT: how the debate momentum shifted\n\n"
        "Each score is an INDEPENDENT integer 0-10 against an absolute-quality rubric. "
        "Do NOT force the scores to sum to any particular total — both can be low (e.g. 3 and 4) "
        "or both can be high (e.g. 8 and 9). If a score is unclear or missing, default it to 5.\n\n"
        f"---\n\n{text[:3000]}"
    )
    try:
        result = run_agent(registry, "eval_parser", prompt, "claude-haiku-4-5-20251001",
                           timeout=30, debate_id=debate_id, debate_dir=debate_dir)
        d = s = 5
        justification = "Evaluation unavailable"
        shift = "N/A"
        for line in result.strip().split("\n"):
            lu = line.upper().strip()
            if lu.startswith("DEFENDER_SCORE") or lu.startswith("DEFENDER SCORE"):
                num = "".join(c for c in line.split(":", 1)[1] if c.isdigit())
                d = max(0, min(10, int(num))) if num else 5
            elif lu.startswith("SKEPTIC_SCORE") or lu.startswith("SKEPTIC SCORE"):
                num = "".join(c for c in line.split(":", 1)[1] if c.isdigit())
                s = max(0, min(10, int(num))) if num else 5
            elif lu.startswith("JUSTIFICATION"):
                justification = line.split(":", 1)[1].strip()
            elif lu.startswith("SHIFT"):
                shift = line.split(":", 1)[1].strip()
        return {"defender_score": d, "skeptic_score": s,
                "defender_confidence": 0.7, "skeptic_confidence": 0.7,
                "evidence_used": [],
                "justification": justification,
                "shift": shift, "reasoning": text[:500] if text else None}
    except Exception:
        log.warning("LLM eval parser failed, defaulting to 5-5")
        return {"defender_score": 5, "skeptic_score": 5,
                "defender_confidence": 0.5, "skeptic_confidence": 0.5,
                "evidence_used": [],
                "justification": "Evaluation unavailable",
                "shift": "N/A", "reasoning": text[:500] if text else None}


def parse_summarizer_response_json(text: str) -> dict | None:
    """Extract the structured verdict JSON from a summarizer or analyst response.

    Returns {verdict, verdict_confidence, key_claims_cited, rationale} or None.
    """
    parsed = extract_json_block(text)
    if not isinstance(parsed, dict):
        return None
    from debater.config import VERDICT_SCALE
    raw_v = str(parsed.get("verdict", "")).strip().upper()
    verdict = next((v for v in VERDICT_SCALE if v == raw_v), None)
    if verdict is None:
        return None
    try:
        conf = max(0.0, min(1.0, float(parsed.get("verdict_confidence", 0.5))))
    except (TypeError, ValueError):
        conf = 0.5
    key_claims = [str(c).strip() for c in (parsed.get("key_claims_cited") or [])
                  if str(c).strip()][:32]
    rationale = str(parsed.get("rationale") or "").strip()[:500]
    return {
        "verdict": verdict,
        "verdict_confidence": conf,
        "key_claims_cited": key_claims,
        "rationale": rationale,
    }


def check_early_stopping(state: dict) -> bool:
    """Check if early stopping condition is met."""
    threshold = state["settings"].get("early_stop_threshold", 3)
    rounds = state.get("rounds", [])
    evaluated = [r for r in rounds if r.get("evaluator")]

    if len(evaluated) < threshold:
        return False

    recent = evaluated[-threshold:]
    d_scores = [r["evaluator"]["defender_score"] for r in recent]
    s_scores = [r["evaluator"]["skeptic_score"] for r in recent]
    return len(set(d_scores)) == 1 and len(set(s_scores)) == 1


def run_round_summarizer(registry: DebateRegistry, state: dict, round_num: int,
                         evaluation: dict, debate_dir: str) -> str:
    """Generate a concise user-friendly summary of a completed debate round."""
    d = evaluation["defender_score"]
    s = evaluation["skeptic_score"]
    winner = "Defender" if d > s else "Skeptic" if s > d else "Tied"

    prior = state["rounds"][:-1]
    prev_line = ""
    if prior and prior[-1].get("evaluator"):
        pe = prior[-1]["evaluator"]
        prev_line = f"\nPrevious round: Defender {pe['defender_score']}, Skeptic {pe['skeptic_score']}"

    prompt = (
        f"A debate about: \"{state['claim'][:120]}\"\n"
        f"Round {round_num} scores — Defender: {d}/10, Skeptic: {s}/10 (leading side: {winner})\n"
        f"Evaluator note: {evaluation['justification']}\n"
        + (f"Evaluator reasoning: {evaluation['reasoning']}\n" if evaluation.get("reasoning") else "")
        + f"Momentum shift: {evaluation['shift']}"
        f"{prev_line}\n\n"
        "Write 1-2 short, plain-English sentences summarising this round for a casual reader. "
        "State who won and the single most important reason why. "
        "Be specific and direct. No markdown, no bullet points, no labels like 'Round X:'."
    )

    models = state["settings"]["models"]
    try:
        summary = run_agent(
            registry,
            "round_summarizer",
            prompt,
            models.get("summarizer", "claude-haiku-4-5-20251001"),
            timeout=30,
            debate_id=state["debate_id"],
            debate_dir=debate_dir,
        )
        return summary.strip()[:400]
    except Exception as e:
        log.warning("Round summarizer failed: %s", e)
        return evaluation["justification"]


def _parse_atomic_claims(response: str, max_per_side: int) -> list[dict]:
    """Stage A parser: extract atomic claim records from the decomposer response."""
    data = extract_json_block(response)
    if not isinstance(data, list):
        return []
    counts = {"defender": 0, "skeptic": 0}
    out: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        side = str(item.get("side", "")).strip().lower()
        if side not in ("defender", "skeptic"):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        if counts[side] >= max_per_side:
            continue
        counts[side] += 1
        side_abbrev = "def" if side == "defender" else "skp"
        cid = str(item.get("claim_id") or
                  f"cl_{side_abbrev}_{counts[side]:03d}").strip()[:32]
        key_terms_raw = item.get("key_terms") or []
        key_terms = [str(t)[:80] for t in key_terms_raw if isinstance(t, (str, int, float))][:6]
        out.append({
            "claim_id": cid,
            "side": side,
            "text": text[:500],
            "key_terms": key_terms,
        })
    return out


def _parse_verification_questions(response: str, max_questions: int) -> list[str]:
    """Stage B parser: pull questions list. Falls back to bullet-line scanning."""
    data = extract_json_block(response)
    if isinstance(data, list):
        questions = [str(q).strip() for q in data if str(q).strip()]
        if questions:
            return questions[:max_questions]
    # Fallback: scan bullet lines.
    questions: list[str] = []
    for raw in response.splitlines():
        line = raw.strip()
        if line.startswith(("-", "*")) or (line[:2].rstrip(".").isdigit() if line else False):
            cleaned = line.lstrip("-*0123456789. ").strip()
            if cleaned:
                questions.append(cleaned)
        if len(questions) >= max_questions:
            break
    return questions[:max_questions]


def _verify_atomic_claim(registry: DebateRegistry, claim_record: dict,
                         state: dict, debate_dir: str, round_data: dict,
                         fc_model: str, researcher_model: str,
                         max_questions: int) -> dict:
    """Stages B + C for one atomic claim: generate questions, search each, verdict."""
    debate_id = state["debate_id"]
    claim_id = claim_record["claim_id"]
    claim_text = claim_record["text"]
    side = claim_record["side"]
    base = {
        "claim_id": claim_id,
        "side": side,
        "text": claim_text,
        "verdict": "UNVERIFIED",
        "confidence": 0.0,
        "evidence_url": "",
        "source_tier": "LOW",
        "rationale": "",
        "questions": [],
    }

    # --- Stage B: verification questions (no shared context) ---
    q_prompt = build_verification_question_prompt(claim_text, max_questions)
    try:
        q_response = run_agent(
            registry, "verifier_questioner", q_prompt, fc_model,
            timeout=60, debate_id=debate_id, debate_dir=debate_dir,
        )
    except AgentError as e:
        base["rationale"] = f"Verification-question generation failed: {e}"
        return base
    questions = _parse_verification_questions(q_response, max_questions)
    if not questions:
        base["rationale"] = "No verification questions generated."
        return base

    # --- Stage C: per-question independent search ---
    qa_pairs: list[dict] = []
    for q in questions:
        try:
            answer = run_researcher(
                registry, q, claim_text, "fact_checker",
                researcher_model, timeout=120,
                debate_id=debate_id, debate_dir=debate_dir,
            )
        except AgentError as e:
            answer = f"[Research failed: {e}]"
        qa_pairs.append({"question": q, "answer": answer})
        round_data.setdefault("research", []).append({
            "requested_by": "fact_checker",
            "query": q,
            "results": answer,
            "timestamp": now_iso(),
        })

    base["questions"] = qa_pairs

    # --- Stage C: per-claim verdict ---
    fc_prompt = build_atomic_fact_check_prompt(claim_text, qa_pairs)
    try:
        fc_response = run_agent(
            registry, "atomic_fact_checker", fc_prompt, fc_model,
            timeout=60, debate_id=debate_id, debate_dir=debate_dir,
        )
    except AgentError as e:
        base["rationale"] = f"Atomic fact-check call failed: {e}"
        return base

    parsed = extract_json_block(fc_response)
    if isinstance(parsed, dict):
        verdict = str(parsed.get("verdict", "")).strip().upper()
        if verdict in _ATOMIC_VERDICTS:
            base["verdict"] = verdict
        try:
            base["confidence"] = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
        except (TypeError, ValueError):
            pass
        base["evidence_url"] = str(parsed.get("evidence_url") or "")[:500]
        tier = str(parsed.get("source_tier", "")).strip().upper()
        if tier in _SOURCE_TIERS:
            base["source_tier"] = tier
        base["rationale"] = str(parsed.get("rationale") or "")[:500]
    else:
        base["rationale"] = "Fact-check verdict was not parseable JSON."
    return base


def _atomic_fact_check_markdown(verified_claims: list[dict]) -> str:
    """Synthesize human-readable markdown from the per-claim JSON records.

    Format is consistent with the legacy fact_checker.md so the evaluator and
    summarizer prompts (which read `round_data['fact_checker']`) continue to
    see directional FALSE/DISPUTED/VERIFIED tags.
    """
    if not verified_claims:
        return "_No atomic claims were extracted from this round._"
    lines: list[str] = []
    for r in verified_claims:
        url = r.get("evidence_url") or ""
        url_part = f" — {url}" if url else ""
        tier = r.get("source_tier", "LOW")
        lines.append(
            f"- **[{r['verdict']}]** `{r['claim_id']}` ({r['side']}, "
            f"conf={r.get('confidence', 0.0):.2f}, tier={tier}): "
            f"{r['text']}{url_part}"
        )
        rationale = r.get("rationale") or ""
        if rationale:
            lines.append(f"  - _Reason:_ {rationale}")
    return "\n".join(lines)


def run_atomic_fact_check(registry: DebateRegistry, state: dict,
                          debate_dir: str, round_data: dict) -> None:
    """Replace the single-pass fact-checker with a decompose → verify → score pipeline.

    Populates `round_data['fact_checker_atomic']` (JSON records) and synthesizes a
    backward-compatible markdown summary into `round_data['fact_checker']` so the
    evaluator and summarizer continue to see directional tags.

    NOTE: currently sequential. With Haiku + concurrency cap=4 and the default
    caps (4 atomic claims/side × 2 questions/claim), this fires ~33 calls/round.
    A ThreadPoolExecutor over the per-claim outer loop is a low-risk follow-up.
    """
    round_num = round_data["round"]
    settings = state["settings"]
    models = settings["models"]
    fc_model = models.get("fact_checker", models.get("evaluator", "claude-haiku-4-5-20251001"))
    researcher_model = models.get("researcher", "claude-haiku-4-5-20251001")
    max_atomic = int(settings.get("max_atomic_claims_per_side", 4))
    max_questions = int(settings.get("max_questions_per_claim", 2))

    defender_text = round_data.get("defender") or ""
    skeptic_text = round_data.get("skeptic") or ""
    if not defender_text.strip() and not skeptic_text.strip():
        round_data["fact_checker_atomic"] = []
        round_data["fact_checker"] = "_No content to fact-check this round._"
        return

    # --- Stage A: decomposition ---
    dec_prompt = build_decomposer_prompt(defender_text, skeptic_text, max_atomic)
    try:
        dec_response = run_agent(
            registry, "decomposer", dec_prompt, fc_model,
            timeout=90, debate_id=state["debate_id"], debate_dir=debate_dir,
        )
    except AgentError as e:
        log.error("[decomposer] Failed: %s", e)
        round_data["fact_checker_atomic"] = []
        round_data["fact_checker"] = f"**[Decomposer failed: {e}]**"
        return

    atomic_claims = _parse_atomic_claims(dec_response, max_atomic)
    debate_log(
        debate_dir,
        f"DECOMPOSER_OUTPUT [R{round_num}]",
        f"{len(atomic_claims)} atomic claims extracted.\n\n{dec_response[:1500]}",
    )

    if not atomic_claims:
        log.warning("[decomposer] No atomic claims extracted in round %d", round_num)
        round_data["fact_checker_atomic"] = []
        round_data["fact_checker"] = "_No atomic claims could be extracted from this round._"
        return

    # --- Stages B + C: per-claim verification (sequential for now) ---
    verified: list[dict] = []
    for claim_record in atomic_claims:
        verdict = _verify_atomic_claim(
            registry, claim_record, state, debate_dir, round_data,
            fc_model, researcher_model, max_questions,
        )
        verified.append(verdict)

    round_data["fact_checker_atomic"] = verified

    # Persist JSON sidecar.
    sidecar_dir = os.path.join(debate_dir, f"round_{round_num}")
    try:
        os.makedirs(sidecar_dir, exist_ok=True)
        with open(os.path.join(sidecar_dir, "fact_checker.json"), "w",
                  encoding="utf-8") as f:
            json.dump(verified, f, indent=2, ensure_ascii=False)
    except OSError as e:
        log.warning("Failed to write fact_checker.json sidecar: %s", e)

    # Synthesize the backward-compatible markdown summary.
    round_data["fact_checker"] = _atomic_fact_check_markdown(verified)
    debate_log(
        debate_dir,
        f"ATOMIC_FACT_CHECK [R{round_num}]",
        f"Verified {len(verified)} atomic claims.\nMarkdown:\n{round_data['fact_checker']}",
    )


def classify_claim(registry: DebateRegistry, claim: str,
                   debate_id: str | None = None,
                   debate_dir: str | None = None) -> str:
    """Pre-classify the claim with Haiku. Returns one of CLAIM_CLASSES.

    Defaults to FACTUAL_TESTABLE on any failure — the safe default that lets the
    normal debate proceed.
    """
    if not claim or not claim.strip():
        return "FACTUAL_TESTABLE"
    try:
        result = run_agent(
            registry,
            "claim_classifier",
            build_claim_classifier_prompt(claim),
            "claude-haiku-4-5-20251001",
            timeout=30,
            debate_id=debate_id,
            debate_dir=debate_dir,
        )
    except Exception as e:
        log.warning("[claim_classifier] Failed (%s) — defaulting to FACTUAL_TESTABLE", e)
        return "FACTUAL_TESTABLE"

    # Find the CLASS: line and extract the token. Falls back to scanning the full
    # response for any one of the four tokens — the prompt is strict but models drift.
    for raw_line in result.splitlines():
        line = raw_line.strip()
        if not line.upper().startswith("CLASS"):
            continue
        if ":" in line:
            payload = line.split(":", 1)[1].strip().upper()
        else:
            payload = line.upper()
        for cls in CLAIM_CLASSES:
            if cls in payload:
                debate_log(debate_dir, "CLAIM_CLASSIFIED",
                           f"Class: {cls}\nClaim: {claim}\nRaw:\n{result[:500]}")
                return cls

    upper = result.upper()
    for cls in CLAIM_CLASSES:
        if cls in upper:
            debate_log(debate_dir, "CLAIM_CLASSIFIED",
                       f"Class: {cls} (fallback scan)\nClaim: {claim}\nRaw:\n{result[:500]}")
            return cls

    log.warning("[claim_classifier] Unparseable response — defaulting to FACTUAL_TESTABLE")
    debate_log(debate_dir, "CLAIM_CLASSIFIER_UNPARSEABLE", result[:1000])
    return "FACTUAL_TESTABLE"


def detect_refusal(registry: DebateRegistry, response: str) -> bool:
    """Use a quick haiku call to determine if the response is a refusal to engage."""
    if not response or len(response) < 20:
        return False
    try:
        result = run_agent(
            registry,
            "refusal_detector",
            f"Is the following text a REFUSAL to engage in the assigned debate role (i.e. the author declines to argue the position)? Answer ONLY YES or NO, nothing else.\n\n{response[:2000]}",
            "claude-haiku-4-5-20251001",
            timeout=15,
        )
        return result.strip().upper().startswith("YES")
    except Exception:
        return False


def generate_title_async(registry: DebateRegistry, debate_dir: str) -> None:
    """Generate a short title for the debate claim (best-effort)."""
    try:
        state = load_state(debate_dir)
        if not state or state.get("title"):
            return
        claim = state["claim"]
        title = run_agent(
            registry,
            "titler",
            f"Compress this claim into its essence in 3-5 words. Return ONLY the short title, nothing else:\n{claim}",
            "claude-haiku-4-5-20251001",
            timeout=30,
        )
        title = title.strip().strip('"\'').strip()
        if title:
            lock = registry.state_lock(state["debate_id"])
            with lock:
                state = load_state(debate_dir)
                state["title"] = title
                save_state(debate_dir, state)
            log.info("[titler] Title: %s", title)
    except Exception as e:
        log.warning("[titler] Failed to generate title: %s", e)


def run_debate_cycle(registry: DebateRegistry, debate_dir: str, state: dict,
                     shutdown_event: threading.Event):
    """Run one complete debate cycle: defender -> skeptic -> fact-checker -> evaluator."""
    round_num = state["current_round"]
    lock = registry.state_lock(state["debate_id"])

    models = state["settings"]["models"]
    efforts = state["settings"]["efforts"]

    round_data = {
        "round": round_num,
        "defender": None,
        "defender_timestamp": None,
        "defender_model": models["defender"],
        "defender_effort": efforts.get("defender"),
        "skeptic": None,
        "skeptic_timestamp": None,
        "skeptic_model": models["skeptic"],
        "skeptic_effort": efforts.get("skeptic"),
        "fact_checker": None,
        "fact_checker_timestamp": None,
        "fact_checker_model": models.get("fact_checker"),
        "fact_checker_effort": efforts.get("fact_checker"),
        "fact_checker_atomic": [],
        "evaluator": None,
        "evaluator_timestamp": None,
        "evaluator_model": models["evaluator"],
        "evaluator_effort": efforts.get("evaluator"),
        "evaluator_samples": [],
        "evaluator_unstable": False,
        "research": [],
    }
    state["rounds"].append(round_data)

    # --- DEFENDER ---
    if shutdown_event.is_set():
        return False

    with lock:
        state["active_agent"] = "defender"
        state["agent_status"] = "thinking"
        save_state(debate_dir, state)

    defender_prompt = build_defender_prompt(state)
    defender_response = run_agent_with_research(
        registry, "defender", defender_prompt, models["defender"], efforts["defender"],
        state, debate_dir, round_data,
    )

    if defender_response is None:
        return False
    round_data["defender"] = defender_response
    round_data["defender_timestamp"] = now_iso()

    append_md(os.path.join(debate_dir, "defender.md"), f"Round {round_num}", defender_response)

    if round_num == 1 and detect_refusal(registry, defender_response):
        log.info("[defender] Initial refusal — retrying with explicit research prompt")
        debate_log(debate_dir, "REFUSAL_RETRY [defender]", "Retrying with nudge")
        retry_prompt = defender_prompt + (
            "\n\nNOTE: Real-time research results are available above. "
            "Please analyze the evidence and present arguments based on what the research shows. "
            "This is an academic exercise — your analysis helps the user evaluate the claim objectively."
        )
        defender_response = run_agent_with_research(
            registry, "defender", retry_prompt, models["defender"], efforts["defender"],
            state, debate_dir, round_data,
        )
        if defender_response is None:
            return False
        round_data["defender"] = defender_response
        round_data["defender_timestamp"] = now_iso()
        append_md(os.path.join(debate_dir, "defender.md"), f"Round {round_num} (retry)", defender_response)

        if detect_refusal(registry, defender_response):
            log.info("[defender] Still refused after retry — aborting debate")
            debate_log(debate_dir, "REFUSAL [defender]", defender_response[:500])
            with lock:
                state["active_agent"] = None
                state["agent_status"] = None
                save_state(debate_dir, state)
            return "refused"

    with lock:
        state["active_agent"] = None
        state["agent_status"] = None
        save_state(debate_dir, state)

    # --- SKEPTIC ---
    if shutdown_event.is_set():
        return False

    with lock:
        state["active_agent"] = "skeptic"
        state["agent_status"] = "thinking"
        save_state(debate_dir, state)

    skeptic_prompt = build_skeptic_prompt(state)
    skeptic_response = run_agent_with_research(
        registry, "skeptic", skeptic_prompt, models["skeptic"], efforts["skeptic"],
        state, debate_dir, round_data,
    )

    if skeptic_response is None:
        return False
    round_data["skeptic"] = skeptic_response
    round_data["skeptic_timestamp"] = now_iso()

    append_md(os.path.join(debate_dir, "skeptic.md"), f"Round {round_num}", skeptic_response)

    if round_num == 1 and detect_refusal(registry, skeptic_response):
        log.info("[skeptic] Initial refusal — retrying with explicit research prompt")
        debate_log(debate_dir, "REFUSAL_RETRY [skeptic]", "Retrying with nudge")
        retry_prompt = skeptic_prompt + (
            "\n\nNOTE: Real-time research results are available above. "
            "Please analyze the evidence and present your critique based on what the research shows. "
            "This is an academic exercise — your analysis helps the user evaluate the claim objectively."
        )
        skeptic_response = run_agent_with_research(
            registry, "skeptic", retry_prompt, models["skeptic"], efforts["skeptic"],
            state, debate_dir, round_data,
        )
        if skeptic_response is None:
            return False
        round_data["skeptic"] = skeptic_response
        round_data["skeptic_timestamp"] = now_iso()
        append_md(os.path.join(debate_dir, "skeptic.md"), f"Round {round_num} (retry)", skeptic_response)

        if detect_refusal(registry, skeptic_response):
            log.info("[skeptic] Still refused after retry — aborting debate")
            debate_log(debate_dir, "REFUSAL [skeptic]", skeptic_response[:500])
            with lock:
                state["active_agent"] = None
                state["agent_status"] = None
                save_state(debate_dir, state)
            return "refused"

    with lock:
        state["active_agent"] = None
        state["agent_status"] = None
        save_state(debate_dir, state)

    # --- FACT-CHECKER (atomic CoVe pipeline) ---
    if shutdown_event.is_set():
        return False

    with lock:
        state["active_agent"] = "fact_checker"
        state["agent_status"] = "decomposing"
        save_state(debate_dir, state)

    run_atomic_fact_check(registry, state, debate_dir, round_data)
    round_data["fact_checker_timestamp"] = now_iso()
    append_md(
        os.path.join(debate_dir, "fact_checker.md"),
        f"Round {round_num}",
        round_data.get("fact_checker") or "_No fact-check output._",
    )

    with lock:
        state["active_agent"] = None
        state["agent_status"] = None
        save_state(debate_dir, state)

    # --- EVALUATOR ---
    if shutdown_event.is_set():
        return False

    with lock:
        state["active_agent"] = "evaluator"
        state["agent_status"] = "thinking"
        save_state(debate_dir, state)

    # Phase 5 self-consistency: run N samples, aggregate by median.
    n_samples = max(1, int(state["settings"].get("evaluator_samples", 3)))
    evaluator_prompt = build_evaluator_prompt(state, round_num)
    samples: list[dict] = []
    for i in range(n_samples):
        try:
            eval_response = run_agent(
                registry, "evaluator", evaluator_prompt,
                models["evaluator"], effort=efforts["evaluator"],
                debate_id=state["debate_id"], debate_dir=debate_dir,
            )
        except AgentError as e:
            log.error("[evaluator] sample %d failed: %s", i + 1, e)
            eval_response = ("SCORE: defender=5, skeptic=5\n"
                             "JUSTIFICATION: Evaluator error, defaulting\nSHIFT: N/A")
        sample = parse_evaluator_response(
            registry, eval_response,
            debate_id=state["debate_id"], debate_dir=debate_dir,
        )
        samples.append(sample)

    evaluation = _aggregate_evaluator_samples(samples)
    round_data["evaluator"] = evaluation
    round_data["evaluator_samples"] = samples
    round_data["evaluator_unstable"] = evaluation.get("evaluator_unstable", False)
    round_data["evaluator_timestamp"] = now_iso()
    debate_log(
        debate_dir, f"EVALUATION [R{round_num}]",
        f"Defender: {evaluation['defender_score']} | "
        f"Skeptic: {evaluation['skeptic_score']} "
        f"(samples={len(samples)}, unstable={evaluation.get('evaluator_unstable', False)})\n"
        f"Justification: {evaluation['justification']}\nShift: {evaluation['shift']}",
    )

    round_data["round_summary"] = run_round_summarizer(registry, state, round_num,
                                                       evaluation, debate_dir)

    append_md(
        os.path.join(debate_dir, "evaluator.md"),
        f"Round {round_num}",
        f"**Defender: {evaluation['defender_score']}** — **Skeptic: {evaluation['skeptic_score']}**\n\n"
        f"{evaluation['justification']}\n\n*Shift: {evaluation['shift']}*",
    )

    with lock:
        state["active_agent"] = None
        state["agent_status"] = None
        save_state(debate_dir, state)

    log.info("Round %d complete: Defender %d - Skeptic %d | %s",
             round_num, evaluation["defender_score"], evaluation["skeptic_score"],
             evaluation["justification"])

    early = check_early_stopping(state)
    if early:
        state["early_stopped"] = True
        log.info("Early stopping triggered: score unchanged for %d rounds",
                 state["settings"]["early_stop_threshold"])

    return early


def run_summarizer(registry: DebateRegistry, debate_dir: str, state: dict) -> None:
    """Run the summarizer agent and save results."""
    lock = registry.state_lock(state["debate_id"])

    with lock:
        state["active_agent"] = "summarizer"
        state["agent_status"] = "thinking"
        save_state(debate_dir, state)

    models = state["settings"]["models"]
    efforts = state["settings"]["efforts"]

    prompt = build_summarizer_prompt(state)
    try:
        summary = run_agent(registry, "summarizer", prompt, models["summarizer"],
                            effort=efforts["summarizer"], debate_id=state["debate_id"],
                            debate_dir=debate_dir)
    except AgentError as e:
        log.error("[summarizer] Failed: %s", e)
        summary = f"**Summarizer error:** {e}\n\nVerdict: INCONCLUSIVE"

    # Phase 3 JSON-first; fall back to LLM extractors on missing/malformed JSON.
    parsed = parse_summarizer_response_json(summary)
    if parsed is not None:
        verdict = parsed["verdict"]
        verdict_confidence = parsed["verdict_confidence"]
        confidence = int(round(verdict_confidence * 100))
        state["verdict_confidence"] = verdict_confidence
        state["key_claims_cited"] = parsed["key_claims_cited"]
    else:
        verdict = extract_verdict_with_llm(registry, summary, debate_id=state["debate_id"], debate_dir=debate_dir)
        confidence = extract_confidence_with_llm(registry, summary, debate_id=state["debate_id"], debate_dir=debate_dir)
        state["verdict_confidence"] = confidence / 100.0 if confidence else None
        state["key_claims_cited"] = []

    state["summary"] = summary
    state["verdict"] = verdict
    state["confidence"] = confidence

    summary_path = os.path.join(debate_dir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# Debate Summary\n\n**Claim:** {state['claim']}\n\n")
        f.write(f"**Verdict:** {verdict}\n\n---\n\n")
        f.write(summary)

    with lock:
        state["active_agent"] = None
        state["agent_status"] = None
        save_state(debate_dir, state)

    log.info("Summary complete. Verdict: %s (conf=%s)", verdict, state.get("verdict_confidence"))


def _aggregate_evaluator_samples(samples: list[dict]) -> dict:
    """Phase 5 self-consistency: median per-side score, mean confidence, instability flag.

    `samples` is a list of dicts produced by `parse_evaluator_response`. Aggregation
    is robust to single-sample input (returns the sample unchanged with
    `evaluator_unstable=False`) and to mixed JSON / fallback samples.
    """
    if not samples:
        # Should not happen — caller always passes ≥1, but defend anyway.
        return {
            "defender_score": 5, "skeptic_score": 5,
            "defender_confidence": 0.5, "skeptic_confidence": 0.5,
            "evidence_used": [],
            "justification": "Evaluation unavailable",
            "shift": "N/A", "reasoning": None,
        }
    if len(samples) == 1:
        out = dict(samples[0])
        out["evaluator_unstable"] = False
        return out

    def _median_int(values: list[int]) -> int:
        sorted_v = sorted(values)
        n = len(sorted_v)
        if n % 2 == 1:
            return sorted_v[n // 2]
        # For even counts, round the average to nearest int (toward even).
        return round((sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2)

    d_scores = [int(s.get("defender_score", 5)) for s in samples]
    s_scores = [int(s.get("skeptic_score", 5)) for s in samples]
    d_confs = [float(s.get("defender_confidence", 0.7)) for s in samples]
    s_confs = [float(s.get("skeptic_confidence", 0.7)) for s in samples]

    unstable = (max(d_scores) - min(d_scores) > 3) or (max(s_scores) - min(s_scores) > 3)

    # Merge evidence_used across samples while preserving order and uniqueness.
    seen: set[str] = set()
    evidence: list[str] = []
    for s in samples:
        for cid in s.get("evidence_used", []):
            if cid and cid not in seen:
                seen.add(cid)
                evidence.append(cid)

    # Pick a representative justification — the sample whose score pair is
    # closest to the aggregate is the most representative.
    d_med = _median_int(d_scores)
    s_med = _median_int(s_scores)

    def _distance(s):
        return abs(int(s.get("defender_score", 5)) - d_med) + abs(int(s.get("skeptic_score", 5)) - s_med)
    representative = min(samples, key=_distance)

    return {
        "defender_score": d_med,
        "skeptic_score": s_med,
        "defender_confidence": sum(d_confs) / len(d_confs),
        "skeptic_confidence": sum(s_confs) / len(s_confs),
        "evidence_used": evidence[:32],
        "justification": representative.get("justification", "see narrative"),
        "shift": representative.get("shift", "see narrative"),
        "reasoning": representative.get("reasoning"),
        "evaluator_unstable": unstable,
    }


def apply_verdict_guard(state: dict) -> None:
    """Deterministic post-summarizer guard against INCONCLUSIVE-default pathology.

    Overrides INCONCLUSIVE → LIKELY TRUE/FALSE only when ALL of the following hold:
    - claim_class == "FACTUAL_TESTABLE"
    - majority of evaluated rounds had a per-side gap of ≥ 2 points
    - the trailing side has ≥ 3 FALSE/DISPUTED atomic claims; leading side has ≤ 1
    For predictive / value / harm-inducing claims the guard is intentionally inert.
    """
    if state.get("claim_class") != "FACTUAL_TESTABLE":
        return
    if state.get("verdict") != "INCONCLUSIVE":
        return

    rounds = [r for r in state.get("rounds", []) if r.get("evaluator")]
    if not rounds:
        return

    gaps: list[int] = []
    for r in rounds:
        e = r["evaluator"]
        try:
            d = int(e.get("defender_score", 5))
            s = int(e.get("skeptic_score", 5))
        except (TypeError, ValueError):
            continue
        gaps.append(s - d)
    if not gaps:
        return

    big_gaps = sum(1 for g in gaps if abs(g) >= 2)
    if big_gaps <= len(gaps) / 2:
        return
    direction = 1 if sum(gaps) > 0 else -1  # +1 = skeptic ahead → claim FALSE

    def _count_false(side: str) -> int:
        return sum(
            1 for r in rounds
            for c in r.get("fact_checker_atomic", [])
            if c.get("side") == side
            and str(c.get("verdict", "")).upper() in ("FALSE", "DISPUTED")
        )

    losing_false = _count_false("defender") if direction > 0 else _count_false("skeptic")
    winning_false = _count_false("skeptic") if direction > 0 else _count_false("defender")

    if losing_false < 3 or winning_false > 1:
        return

    new_verdict = "LIKELY FALSE" if direction > 0 else "LIKELY TRUE"
    log.info(
        "[verdict_guard] Overriding INCONCLUSIVE → %s (gaps_big=%d/%d, "
        "losing_false=%d, winning_false=%d)",
        new_verdict, big_gaps, len(gaps), losing_false, winning_false,
    )
    state["verdict"] = new_verdict
    state["verdict_guard_applied"] = True
    state["verdict_guard_reason"] = (
        f"FACTUAL_TESTABLE claim with majority score-gap≥2 ({big_gaps}/{len(gaps)} rounds), "
        f"losing_false={losing_false}, winning_false={winning_false}; "
        f"direction={'skeptic-ahead' if direction > 0 else 'defender-ahead'}"
    )


def run_analysis_mode(registry: DebateRegistry, debate_dir: str, state: dict,
                      entry_reason: str = "refusal") -> None:
    """Run single-analyst mode when adversarial debate is skipped.

    `entry_reason` is recorded on state and adjusts the analyst's framing:
      - "refusal": agents refused mid-debate (legacy default)
      - "classifier": pre-debate classifier routed a HARM_INDUCING_FALSEHOOD here
    """
    lock = registry.state_lock(state["debate_id"])
    log.info("[analyst] ENTER mode=%s claim=%s", entry_reason, state["claim"][:60])
    debate_log(debate_dir, "ANALYSIS_MODE_ENTER",
               f"entry_reason: {entry_reason}\nclaim: {state['claim']}")

    with lock:
        state["status"] = "analyzing"
        state["active_agent"] = "analyst"
        state["agent_status"] = "analyzing"
        state["analysis_entry_reason"] = entry_reason
        save_state(debate_dir, state)

    models = state["settings"]["models"]
    efforts = state["settings"]["efforts"]
    round_data = {"round": 0, "research": []}

    prompt = build_analyst_prompt(state["claim"], entry_reason=entry_reason)
    debate_log(debate_dir, "ANALYSIS_AGENT_START", f"model={models.get('analyst')}")
    try:
        analysis = run_agent_with_research(
            registry, "analyst", prompt, models.get("analyst", "claude-haiku-4-5-20251001"),
            efforts.get("analyst"), state, debate_dir, round_data,
        )
    except Exception as e:
        log.exception("[analyst] Failed: %s", e)
        debate_log(debate_dir, "ANALYSIS_AGENT_ERROR", f"{type(e).__name__}: {e}")
        analysis = f"**[Analysis failed: {e}]**"

    if analysis is None:
        log.warning("[analyst] returned None — recording as interrupted")
        debate_log(debate_dir, "ANALYSIS_AGENT_INTERRUPTED", "")
        analysis = "Analysis was interrupted."
    debate_log(debate_dir, "ANALYSIS_AGENT_RETURNED", f"Length: {len(analysis)} chars")

    state["analysis"] = analysis

    with open(os.path.join(debate_dir, "analysis.md"), "w", encoding="utf-8") as f:
        f.write(analysis)

    # Phase 3 JSON-first verdict extraction; fall back to LLM extractors.
    parsed = parse_summarizer_response_json(analysis)
    if parsed is not None:
        verdict = parsed["verdict"]
        verdict_confidence = parsed["verdict_confidence"]
        confidence = int(round(verdict_confidence * 100))
        state["verdict_confidence"] = verdict_confidence
    else:
        verdict = extract_verdict_with_llm(registry, analysis, debate_id=state["debate_id"], debate_dir=debate_dir)
        confidence = extract_confidence_with_llm(registry, analysis, debate_id=state["debate_id"], debate_dir=debate_dir)
        state["verdict_confidence"] = confidence / 100.0 if confidence else None

    with lock:
        state["verdict"] = verdict
        state["confidence"] = confidence
        state["summary"] = analysis
        state["status"] = "complete"
        state["active_agent"] = None
        state["agent_status"] = None
        save_state(debate_dir, state)

    note = (
        "This claim was pre-classified as a HARM-INDUCING FALSEHOOD; the adversarial "
        "debate was skipped and an analyst produced a direct evidence-based assessment."
        if entry_reason == "classifier"
        else "This claim was analyzed in single-analyst mode because debate agents declined to engage."
    )
    summary_path = os.path.join(debate_dir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"*Note: {note}*\n\n---\n\n{analysis}")

    log.info("[analyst] EXIT verdict=%s", verdict)
    debate_log(debate_dir, "ANALYSIS_MODE_EXIT",
               f"verdict: {verdict}\nconfidence: {confidence}\nentry_reason: {entry_reason}")


def run_full_debate(registry: DebateRegistry, debate_dir: str,
                    shutdown_event: threading.Event) -> None:
    """Background thread: run all cycles of a debate."""
    state = load_state(debate_dir)
    if not state:
        log.error("Cannot load state from %s", debate_dir)
        return

    debate_id = state["debate_id"]
    lock = registry.state_lock(debate_id)

    with lock:
        state["status"] = "running"
        save_state(debate_dir, state)

    max_iter = state["settings"]["max_iterations"]
    state["rounds"] = [r for r in state.get("rounds", []) if r.get("evaluator")]
    completed_rounds = len(state["rounds"])
    start = completed_rounds + 1

    log.info("Debate '%s' starting from round %d (max %d)", state["claim"][:50], start, max_iter)
    debate_log(debate_dir, "DEBATE_START" if start == 1 else "DEBATE_RESUME",
               f"Claim: {state['claim']}\nRounds: {start} to {max_iter}\n"
               f"Models: {json.dumps(state['settings']['models'], indent=2)}\n"
               f"Efforts: {json.dumps(state['settings']['efforts'], indent=2)}\n"
               f"Prior debates: {len(state.get('prior_debates', []))}")

    # Pre-debate claim classifier (only on fresh start, not on resume).
    if start == 1 and not state.get("claim_class"):
        with lock:
            state["active_agent"] = "claim_classifier"
            state["agent_status"] = "classifying"
            save_state(debate_dir, state)
        claim_class = classify_claim(
            registry, state["claim"],
            debate_id=debate_id, debate_dir=debate_dir,
        )
        with lock:
            state["claim_class"] = claim_class
            state["active_agent"] = None
            state["agent_status"] = None
            save_state(debate_dir, state)
        log.info("Claim classified as %s", claim_class)

        if claim_class == "HARM_INDUCING_FALSEHOOD":
            log.info("Skipping adversarial debate — routing directly to analyst")
            debate_log(debate_dir, "CLASSIFIER_ROUTED_TO_ANALYST",
                       f"Class: {claim_class}\nClaim: {state['claim']}")
            try:
                run_analysis_mode(registry, debate_dir, state, entry_reason="classifier")
            except Exception as e:
                log.exception("Analyst (from classifier) failed: %s", e)
                debate_log(debate_dir, "ANALYSIS_MODE_FAILED",
                           f"entry_reason=classifier\n{type(e).__name__}: {e}")
                with lock:
                    state["status"] = "analysis_failed"
                    state["active_agent"] = None
                    state["agent_status"] = str(e)
                    save_state(debate_dir, state)
            registry.unregister_debate(debate_id)
            log.info("Debate '%s' finished with status: %s",
                     state["claim"][:50], state["status"])
            debate_log(debate_dir, "DEBATE_COMPLETE",
                       f"Status: {state['status']}\nRoute: classifier→analyst\n"
                       f"Verdict: {state.get('verdict', 'N/A')}")
            return

    if start == 1 and not state.get("pre_research"):
        log.info("Pre-researching claim before debate starts")
        with lock:
            state["active_agent"] = "researcher"
            state["agent_status"] = "researching"
            save_state(debate_dir, state)
        try:
            pre_research_claim(registry, state, debate_dir)
        except Exception as e:
            log.warning("Pre-research failed (continuing without): %s", e)
        with lock:
            state["active_agent"] = None
            state["agent_status"] = None
            save_state(debate_dir, state)

    try:
        for cycle in range(start, max_iter + 1):
            if shutdown_event.is_set():
                log.info("Debate stopped by user")
                break

            user_msgs = state.get("user_messages", [])
            last_seen = state.get("last_seen_user_msg_idx", 0)
            if len(user_msgs) > last_seen:
                log.info("New user interjections: %d", len(user_msgs) - last_seen)

            state["current_round"] = cycle
            with lock:
                save_state(debate_dir, state)

            result = run_debate_cycle(registry, debate_dir, state, shutdown_event)

            if result == "refused":
                log.info("Debate refused by agent — auto-triggering analyst mode")
                debate_log(debate_dir, "DEBATE_REFUSED", "Agent declined to engage — switching to analyst mode automatically")
                try:
                    run_analysis_mode(registry, debate_dir, state, entry_reason="refusal")
                except Exception as e:
                    log.exception("Analyst (from refusal) failed: %s", e)
                    debate_log(debate_dir, "ANALYSIS_MODE_FAILED",
                               f"entry_reason=refusal\n{type(e).__name__}: {e}")
                    with lock:
                        state["status"] = "analysis_failed"
                        state["active_agent"] = None
                        state["agent_status"] = str(e)
                        save_state(debate_dir, state)
                break

            state["last_seen_user_msg_idx"] = len(state.get("user_messages", []))
            with lock:
                save_state(debate_dir, state)

            if result:
                break

        # Final summarizer pass — only if nothing else has already finalized status.
        terminal_statuses = ("refused", "complete", "analysis_failed", "error")
        if not shutdown_event.is_set() and state.get("status") not in terminal_statuses:
            run_summarizer(registry, debate_dir, state)
            apply_verdict_guard(state)
            with lock:
                state["status"] = "complete"
                save_state(debate_dir, state)
        else:
            with lock:
                if state.get("status") not in terminal_statuses:
                    state["status"] = "stopped"
                state["active_agent"] = None
                state["agent_status"] = None
                save_state(debate_dir, state)

    except Exception as e:
        log.exception("Debate failed with exception: %s", e)
        with lock:
            state["status"] = "error"
            state["active_agent"] = None
            state["agent_status"] = str(e)
            save_state(debate_dir, state)

    registry.unregister_debate(debate_id)

    log.info("Debate '%s' finished with status: %s", state["claim"][:50], state["status"])
    evaluated = [r for r in state.get("rounds", []) if r.get("evaluator")]
    debate_log(debate_dir, "DEBATE_COMPLETE",
               f"Status: {state['status']}\nRounds completed: {len(evaluated)}\n"
               f"Verdict: {state.get('verdict', 'N/A')}\n"
               f"Early stopped: {state.get('early_stopped', False)}")
