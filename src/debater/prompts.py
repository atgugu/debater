"""Prompt templates and context-assembly helpers. Pure functions."""
from __future__ import annotations

import re

from debater.config import CLAIM_CLASSES, VERDICT_SCALE


def tail_sections(content: str, n: int, section_marker: str = "## ") -> str:
    """Return file header + last n sections."""
    if not content or n <= 0:
        return content
    pattern = re.compile(r"(?=^" + re.escape(section_marker) + r")", re.MULTILINE)
    parts = pattern.split(content)
    if len(parts) <= 1:
        return content
    header = parts[0]
    sections = parts[1:]
    if len(sections) <= n:
        return content
    return header + "".join(sections[-n:])


def build_agent_context(state: dict, agent_name: str) -> str:
    """Build context string for an agent's prompt."""
    claim = state["claim"]
    rounds = state.get("rounds", [])
    opponent = "skeptic" if agent_name == "defender" else "defender"

    # Collect own responses
    own_parts = []
    for r in rounds:
        if r.get(agent_name):
            own_parts.append((r["round"], r[agent_name]))

    # Collect opponent's last response
    opponent_last = ""
    for r in reversed(rounds):
        if r.get(opponent):
            text = r[opponent]
            if len(text) > 8000:
                text = text[:8000] + "\n[... truncated ...]"
            opponent_last = f"OPPONENT'S LATEST ARGUMENT (Round {r['round']}):\n{text}"
            break

    # Build own history with condensation
    total_own_chars = sum(len(t) for _, t in own_parts)
    if total_own_chars > 15000 and len(own_parts) > 2:
        lines = []
        first_text = own_parts[0][1][:2000] + "\n[... condensed ...]"
        lines.append(f"Your Round {own_parts[0][0]} (condensed):\n{first_text}")
        for rnum, text in own_parts[1:-2]:
            lines.append(f"Your Round {rnum}: [You argued: {text[:200]}...]")
        for rnum, text in own_parts[-2:]:
            lines.append(f"Your Round {rnum} (full):\n{text}")
        own_history = "\n\n".join(lines)
    else:
        own_history = "\n\n".join(
            f"Your Round {rnum}:\n{text}" for rnum, text in own_parts
        )

    # Evaluator scores (compact)
    score_lines = []
    for r in rounds:
        if r.get("evaluator"):
            e = r["evaluator"]
            score_lines.append(
                f"Round {r['round']}: Defender {e['defender_score']} - "
                f"Skeptic {e['skeptic_score']} ({e['justification']})"
            )
    scores_text = "\n".join(score_lines) if score_lines else "(No scores yet)"

    # User interjections (unseen)
    interjections = ""
    user_msgs = state.get("user_messages", [])
    last_seen = state.get("last_seen_user_msg_idx", 0)
    unseen = user_msgs[last_seen:]
    if unseen:
        interjection_parts = [
            f"- [During Round {m.get('during_round', '?')}, while {m.get('during_agent', '?')} was active] {m['message']}"
            for m in unseen
        ]
        interjections = "USER INTERJECTIONS (you MUST acknowledge and address these):\n" + "\n".join(interjection_parts)

    # Research results (last 5, attributed) — include pre-research
    research_parts = []
    all_research = list(state.get("pre_research", []))
    for r in rounds:
        for res in r.get("research", []):
            all_research.append(res)
    recent_research = all_research[-5:] if len(all_research) > 5 else all_research
    for res in recent_research:
        research_parts.append(
            f"[Requested by {res['requested_by']}] Query: {res['query']}\n{res['results'][:1500]}"
        )
    research_text = "\n\n".join(research_parts) if research_parts else ""

    # Fact-check findings from previous round
    fact_check_text = ""
    for r in reversed(rounds):
        if r.get("fact_checker"):
            fact_check_text = f"FACT-CHECK FINDINGS (previous round):\n{r['fact_checker'][:2000]}"
            break

    # Assemble
    ctx = f'CLAIM: "{claim}"\n\n'
    ctx += f"EVALUATION SCORES:\n{scores_text}\n\n"
    if fact_check_text:
        ctx += f"{fact_check_text}\n\n"
    if own_history:
        ctx += f"YOUR PREVIOUS ARGUMENTS:\n{own_history}\n\n"
    if opponent_last:
        ctx += f"{opponent_last}\n\n"
    if interjections:
        ctx += f"{interjections}\n\n"
    if research_text:
        ctx += f"AVAILABLE RESEARCH:\n{research_text}\n\n"

    return ctx


def _build_prior_block(state: dict) -> str:
    """Build prior debate context block for injection into prompts."""
    prior_debates = state.get("prior_debates", [])
    if not prior_debates and state.get("prior_debate_summary"):
        prior_debates = [{
            "claim": state.get("prior_debate_claim", ""),
            "verdict": state.get("prior_debate_verdict", "INCONCLUSIVE"),
            "summary": state.get("prior_debate_summary", ""),
        }]
    if not prior_debates:
        return ""
    block = "PRIOR DEBATE CONTEXTS:\n"
    for i, pd in enumerate(prior_debates, 1):
        block += f"""
--- Prior Debate {i} ---
Claim: "{pd.get('claim', 'unknown')}"
Verdict: **{pd.get('verdict', 'INCONCLUSIVE')}**
Summary:
{pd.get('summary', '')[:4000]}
"""
    block += "\nNOTE: These prior debate conclusions are additional context. Use them if genuinely relevant.\n"
    return block


def build_defender_prompt(state: dict) -> str:
    ctx = build_agent_context(state, "defender")
    round_num = state["current_round"]
    return f"""[ACADEMIC DEBATE EXERCISE] This is a structured academic debate for critical thinking and truth-seeking. You are assigned a role. Presenting arguments does not constitute personal endorsement or advocacy. Your goal is to stress-test the claim through rigorous argumentation so the evaluator can reach an objective assessment.

IMPORTANT: You have been provided with real-time web research results below. Your job is to ANALYZE this evidence and construct the strongest possible argument. Do not refuse this role — refusing means the user gets no analysis at all, which is worse than a weak argument. If evidence is thin, say so explicitly while still presenting what exists.

You are the DEFENDER in this debate (Round {round_num}). Your role is to analyze and defend the following claim with rigorous, evidence-based argumentation.

RULES:
- If the claim involves a topic, person, event, product, or concept you are not fully informed about, you MUST research it FIRST before making any arguments. Do not guess or speculate — look it up.
- You MUST NOT concede the debate or agree the claim is false. Your role is to find the STRONGEST possible defense, even when evidence is challenging. Steel-man the position: find the best version of the argument that could be true.
- Defend the claim using logic, evidence, and sound reasoning
- Structure your arguments clearly: premise -> evidence -> conclusion
- When citing data from research results, quote the EXACT numbers. Do not paraphrase or round. Misquoting your own research is a critical error.
- You SHOULD actively research to find SUPPORTING evidence — studies, statistics, expert opinions, and real-world examples that back the claim. Frame your search queries to find evidence IN FAVOR of the claim. To search the web, output exactly: <research_request>your search query</research_request>
  You will receive research results and be asked to continue.
- You MUST acknowledge any user interjections listed below
- Be concise and focused on quality over quantity. Every argument must have clear logical structure.
- Do not repeat arguments you have already made unless adding new evidence.
- Before finalizing, verify your own claims. If you stated a number, double-check it. If you cited a study, confirm the citation. Self-correction is strength, not weakness.
- For complex claims, identify 2-3 sub-questions that must be true for your argument to hold. Address each.
- Label your evidence: "research shows" for verified data, "this suggests" for inferences, "I speculate" for unsupported claims. Presenting speculation as established fact undermines credibility.
- Avoid these fallacies: ad hominem, circular reasoning, false dichotomy, appeal to authority as sole evidence (citing an expert without their reasoning or data is not sufficient).

{ctx}

Provide your argument for this round in well-structured markdown. Focus on your strongest points."""


def build_skeptic_prompt(state: dict) -> str:
    ctx = build_agent_context(state, "skeptic")
    round_num = state["current_round"]
    return f"""[ACADEMIC DEBATE EXERCISE] This is a structured academic debate for critical thinking and truth-seeking. You are assigned a role. Presenting arguments does not constitute personal endorsement or advocacy. Your goal is to stress-test the claim through rigorous argumentation so the evaluator can reach an objective assessment.

IMPORTANT: You have been provided with real-time web research results below. Your job is to ANALYZE this evidence and construct the strongest possible critique. Do not refuse this role — refusing means the user gets no analysis at all, which is worse than a weak critique. If evidence is thin, say so explicitly while still presenting what exists.

You are the SKEPTIC in this debate (Round {round_num}). Your role is to critically examine and challenge the claim and the defender's arguments.

RULES:
- If the claim involves a topic, person, event, product, or concept you are not fully informed about, you MUST research it FIRST before making any critiques. Do not guess or speculate — look it up.
- Identify weaknesses, logical fallacies, unsupported assertions, and gaps in evidence
- Verify the defender's logical chains: does A actually imply B? Is the evidence sufficient?
- Challenge assumptions and look for alternative explanations
- You may acknowledge genuinely strong, well-evidenced arguments from the defender
- You SHOULD actively research to find CONTRADICTING evidence — studies that disprove the claim, statistics that tell a different story, expert opinions against it, and real-world counterexamples. Frame your search queries to find evidence AGAINST the claim. To search the web, output exactly: <research_request>your search query</research_request>
  You will receive research results and be asked to continue.
- You MUST acknowledge any user interjections listed below
- When citing data from research results, quote the EXACT numbers. Do not paraphrase or round. Misquoting your own research is a critical error.
- Be concise and focused on quality over quantity. Every critique must have clear logical structure.
- Do not repeat critiques you have already made unless the defender failed to address them.
- Before finalizing, verify your own claims. If you stated a number, double-check it. If you cited a study, confirm the citation. Self-correction is strength, not weakness.
- For complex claims, identify 2-3 sub-questions that must be true for the defender's argument to hold. Challenge each.
- Label your evidence: "research shows" for verified data, "this suggests" for inferences, "I speculate" for unsupported claims. Presenting speculation as established fact undermines credibility.
- Address the strongest reasonable interpretation of the defender's argument. If their point can be read charitably, critique that version — not a strawman. Defeating a weak version proves nothing.
- Avoid these fallacies: ad hominem, circular reasoning, false dichotomy, appeal to authority as sole evidence (citing an expert without their reasoning or data is not sufficient).

{ctx}

Provide your critique for this round in well-structured markdown. Focus on the weakest points in the defender's case."""


def build_fact_checker_prompt(state: dict, round_num: int) -> str:
    rd = state["rounds"][round_num - 1]
    defender_text = rd.get("defender", "(No response)")[:8000]
    skeptic_text = rd.get("skeptic", "(No response)")[:8000]
    claim = state["claim"]
    prior_block = _build_prior_block(state)

    return f"""You are a FACT-CHECKER in a structured debate (Round {round_num}).

CRITICAL CONTEXT: Both the Defender and Skeptic are LLMs. They may hallucinate — fabricating statistics, misattributing quotes, citing non-existent studies, or inventing data. Trust NOTHING at face value. Verify everything.

YOUR TASK:
Extract the key factual claims made by both sides below, then verify each one by searching the web.

IMPORTANT: You DO have web search access. To search, output this exact tag anywhere in your response:
<research_request>your search query here</research_request>
The system will execute the search and return results to you. You will then be asked to continue with the results. This is NOT a regular tool call — just output the XML tag as text and the system handles it. You can and SHOULD use multiple search tags.

RULES:
- You MUST search the web to verify factual claims, prioritizing the most impactful ones first. Do NOT skip research or say you cannot search — you CAN. Just output <research_request>query</research_request> tags and you will receive results.
- Search multiple times per claim if needed: try different query phrasings, check primary sources, cross-reference numbers. Cast a wide net.
- Prioritize highly trustworthy sources: peer-reviewed journals, official government statistics, established news outlets (Reuters, AP, BBC), academic institutions, WHO/UN/World Bank data
- For each claim, assign one of these confidence levels:
  - **VERIFIED** — confirmed by multiple credible sources
  - **LIKELY TRUE** — supporting evidence found, minor gaps
  - **UNVERIFIED** — insufficient evidence to confirm or deny
  - **DISPUTED** — contradicting evidence exists
  - **FALSE** — clearly disproven by evidence
- Check MATHEMATICAL CALCULATIONS, not just raw facts. If an agent says '67,000 + 65,000 + 27,000 = 159,000 total votes' — verify each number AND the arithmetic.
- Be DIRECT. Do not call problems 'framing disputes.' State clearly which side's evidence is stronger and why. Your job is truth, not diplomacy.
- Be concise. One line per claim: the claim, the verdict, and the key source.
- Focus on the most important/impactful claims, not trivial ones.
- Do NOT assume a claim is true because it cites a source. Citations can be fabricated, misattributed, or taken out of context. Verify the actual content, not the citation.
- Be equally skeptical of confident and tentative claims. Confidence in delivery ≠ factual accuracy. A claim stated with certainty still needs verification.

CLAIM UNDER DEBATE:
{claim}

DEFENDER'S ARGUMENTS (Round {round_num}):
{defender_text}

SKEPTIC'S ARGUMENTS (Round {round_num}):
{skeptic_text}

{prior_block}
Output a concise markdown list of fact-checked claims. Format each as:
- **[VERDICT]** Claim text — Source/evidence (one line)"""


def build_evaluator_prompt(state: dict, round_num: int) -> str:
    rd = state["rounds"][round_num - 1]
    defender_text = rd.get("defender", "(No response)")
    skeptic_text = rd.get("skeptic", "(No response)")

    score_history = ""
    for r in state["rounds"][:round_num - 1]:
        if r.get("evaluator"):
            e = r["evaluator"]
            score_history += f"Round {r['round']}: Defender {e['defender_score']} - Skeptic {e['skeptic_score']} ({e['justification']})\n"

    prior_block = _build_prior_block(state)

    return f"""You are the EVALUATOR in a structured debate. Impartially assess the quality of arguments from both sides in Round {round_num}.

CLAIM: "{state['claim']}"

DEFENDER'S ARGUMENT (Round {round_num}):
{defender_text[:10000]}

SKEPTIC'S ARGUMENT (Round {round_num}):
{skeptic_text[:10000]}

{f"PREVIOUS SCORES:{chr(10)}{score_history}" if score_history else "This is the first round."}
{prior_block}
{f"FACT-CHECK RESULTS (independent verification of claims made this round):{chr(10)}{rd.get('fact_checker', '')[:6000]}{chr(10)}{chr(10)}CRITICAL: Use these fact-check findings to HEAVILY inform your scoring. Agents making VERIFIED claims score higher. Agents citing FALSE or DISPUTED data MUST be penalized 2-3 points — factual accuracy is non-negotiable.{chr(10)}" if rd.get('fact_checker') else ""}
EVALUATION CRITERIA:
- Logical validity: Are the reasoning chains sound?
- Evidence quality: Are claims backed by evidence? Is evidence reliable? (Consider the fact-check results above)
- Argument strength: Are arguments compelling and well-structured?
- Objectivity: Is the agent being fair and analytical?
- Responsiveness: Does the agent address the opponent's points?
- Claim fidelity: Does the defender maintain the original claim's strength ('will happen') or weaken it ('could plausibly happen')? Weakening the claim standard is a significant penalty.
- Factual accuracy: CRITICAL — if the fact-checker found FALSE or DISPUTED claims from either side, this MUST significantly impact scoring. An agent citing false data should lose 2-3 points regardless of logical quality.

DEBIASING GUIDELINES (apply before scoring):
- POSITION: The defender appears first and skeptic second — this order does NOT reflect quality. Evaluate each independently.
- VERBOSITY: Argument quality ≠ argument length. A concise, well-evidenced point scores higher than a verbose, repetitive one. Do not reward padding.
- CONFIDENCE: Be skeptical of confident-sounding assertions. Bold delivery ≠ strong evidence. An uncertain claim backed by data outweighs a confident claim without sources.
- RHETORIC: Evaluate logical content separately from rhetorical style. Emotional appeal, vivid metaphors, or authoritative tone are not evidence.
- INDEPENDENCE: Your score must be based solely on this round's arguments. Ignore cumulative momentum — a side that has been losing can make a strong argument.

SCORING — INDEPENDENT 0–10 RUBRIC:
Score each side INDEPENDENTLY against this absolute-quality scale. The two scores do NOT need to sum to any particular number. It is fully valid for both sides to score low (e.g. defender=3, skeptic=4) when both are weak, or for both to score high (e.g. defender=8, skeptic=9) when both are excellent.

- **9–10** Excellent — rigorous logic, multiple verified sources, addresses counterarguments, zero fact-check failures.
- **7–8** Strong — sound reasoning, evidence present and mostly accurate, addresses opponent's main points.
- **5–6** Adequate — reasonable structure but partially unevidenced, or one significant gap in logic/coverage.
- **3–4** Weak — significant gaps, unsupported assertions, or minor fact-check failures (DISPUTED/UNVERIFIED claims).
- **1–2** Very poor — major logical errors, fabricated or FALSE claims per fact-checker, near-refusal, or evasion.
- **0** No meaningful response — refusal, empty, or completely off-topic.

IMPORTANT: Evaluate this round on its own merits. Do NOT anchor to previous round scores. Do NOT default to symmetric scores like 5/5 out of caution — give the score each side actually earned against the rubric. If one side clearly dominates, that's fine; if both are weak, give both low scores.

Respond in this EXACT format:
REASONING: For each criterion above, note which side was stronger (1-2 words each). Then state your score.
SCORE: defender=N, skeptic=M
JUSTIFICATION: one-line explanation of the score
SHIFT: how the debate shifted since last round, or "First round" if round 1

After the SHIFT line, emit EXACTLY ONE fenced ```json block as the final element of your response. The block must contain:

```json
{{
  "defender": {{"score": <0-10 int>, "confidence": <0.0-1.0 float>, "rationale": "<one sentence>"}},
  "skeptic":  {{"score": <0-10 int>, "confidence": <0.0-1.0 float>, "rationale": "<one sentence>"}},
  "evidence_used": ["<claim_id>", ...]
}}
```

`confidence` is YOUR confidence in the score you assigned (NOT the agent's confidence). 0.9+ means you are very sure about this score; 0.5 means a genuine coin flip; lower is rare. `evidence_used` is a list of atomic-claim IDs (from the FACT-CHECK RESULTS above, format `cl_def_001` / `cl_skp_001`) that materially shaped your scoring. Empty list if none applied. Nothing after the closing fence."""


def build_summarizer_prompt(state: dict) -> str:
    prior_block = _build_prior_block(state)

    record = f'CLAIM: "{state["claim"]}"\n\n'
    for r in state["rounds"]:
        record += f"=== ROUND {r['round']} ===\n\n"
        if r.get("defender"):
            record += f"DEFENDER:\n{r['defender']}\n\n"
        if r.get("skeptic"):
            record += f"SKEPTIC:\n{r['skeptic']}\n\n"
        if r.get("fact_checker"):
            record += f"FACT-CHECK:\n{r['fact_checker'][:2000]}\n\n"
        if r.get("evaluator"):
            e = r["evaluator"]
            record += f"EVALUATION: Defender {e['defender_score']} - Skeptic {e['skeptic_score']} ({e['justification']})\n\n"
        for res in r.get("research", []):
            record += f"RESEARCH ({res['requested_by']}): {res['query']}\n{res['results'][:500]}\n\n"

    for msg in state.get("user_messages", []):
        record += f"USER INTERJECTION: {msg['message']}\n\n"

    if len(record) > 80000:
        record = record[:80000] + "\n[... truncated ...]"

    verdict_options = " | ".join(VERDICT_SCALE)

    return f"""You are the SUMMARIZER for a structured debate. Analyze the complete debate record and produce a final assessment.

{record}

ANALYSIS GUIDELINES:
- Weight all rounds equally. Later rounds are not inherently more important than earlier ones. A strong argument in Round 1 that was never refuted carries more weight than a weak argument in the final round.
- Judge argument quality, not argument length. A side that wrote more is not necessarily a side that argued better.
- Separate rhetorical skill from evidential strength. A well-written argument with no evidence is weaker than a plainly stated argument backed by facts.

Produce your summary with these sections:

### Strongest Arguments For the Claim
Rank the defender's arguments from strongest to weakest. Start with the most compelling, best-evidenced arguments first. For each, note whether the skeptic effectively challenged it or it stood unchallenged.

### Strongest Arguments Against the Claim
Rank the skeptic's arguments from strongest to weakest. Start with the most damaging, best-evidenced critiques first. For each, note whether the defender successfully rebutted it or it stood unchallenged.

### Key Evidence
The most impactful evidence cited by either side, ranked by reliability and relevance. Note which side it favors.

### Weakest Arguments
The weakest arguments from each side, ranked from worst to least bad. Explain why each failed (flawed logic, no evidence, irrelevant, etc).

### Assessment
How the debate evolved, which side gained ground, where the balance settled.

### Comparison to Original Claim
How does the nuanced picture compare to the user's original claim? What aspects are supported vs unsupported?

{prior_block}
### Verdict
**VERDICT_HERE**
Brief explanation of why this verdict was reached, including confidence level.

IMPORTANT VERDICT GUIDANCE:
- For claims about PAST or PRESENT facts (verifiable): use the full range freely. "REFUTED" means disproven by evidence.
- For claims about FUTURE events (predictions): be more cautious with extremes. A future prediction cannot be "REFUTED" or "CONFIRMED" — it hasn't happened yet. Use "LIKELY FALSE" / "STRONGLY REFUTED" for predictions that lack mechanisms and evidence, or "LIKELY TRUE" / "STRONGLY SUPPORTED" for well-evidenced predictions. Reserve "REFUTED"/"CONFIRMED" only for claims that are already falsified/verified by events that have occurred.

VERDICT must be exactly one of: {verdict_options}
Replace VERDICT_HERE with the chosen verdict wrapped in ** markers.

After the verdict narrative, emit EXACTLY ONE fenced ```json block as the final element of your response. The block must contain:

```json
{{
  "verdict": "<one of: {verdict_options}>",
  "verdict_confidence": <0.0-1.0 float>,
  "key_claims_cited": ["<atomic claim_id>", ...],
  "rationale": "<one sentence summarizing why this verdict>"
}}
```

`verdict_confidence` is YOUR confidence in the verdict (not the debaters' confidence). 0.9+ means very sure; ≈0.5 is a coin flip; INCONCLUSIVE verdicts should typically carry confidence ≤ 0.5. `key_claims_cited` is a list of atomic-claim IDs (format `cl_def_001` / `cl_skp_001`) from the fact-check records above that decisively shaped the verdict. Empty list if none applied. Nothing after the closing fence."""


def build_analyst_prompt(claim: str, entry_reason: str = "refusal") -> str:
    if entry_reason == "classifier":
        opener = (
            "You are an objective ANALYST. The following claim was pre-classified as a "
            "HARM-INDUCING FALSEHOOD (well-established false claim whose adversarial defense "
            "would risk amplifying misinformation). Adversarial debate was skipped intentionally; "
            "your job is to give the user the same rigorous evidence-based answer they would "
            "have gotten from a completed debate, without staging the harmful argument."
        )
    else:
        opener = (
            "You are an objective ANALYST. The following claim was submitted for structured "
            "debate, but the debate agents declined to engage in adversarial format due to the "
            "sensitivity of the topic."
        )
    return f"""{opener}

Your task: provide a thorough, balanced, evidence-based analysis of this claim. You MUST research this topic online before forming conclusions.

IMPORTANT: You DO have web search access. To search, output this exact tag:
<research_request>your search query here</research_request>
The system will return results. Use multiple searches to build a comprehensive picture.

CLAIM: "{claim}"

Structure your analysis as follows:

### Why This Claim Exists
Why do some people believe this? What are the origins and key proponents?

### Scientific / Factual Consensus
What does the weight of evidence say? Cite major studies, meta-analyses, official positions.

### Key Evidence
The most important evidence on all sides, ranked by quality and reliability.

### Common Misconceptions
What gets misunderstood or misrepresented about this topic?

### Verdict
Your objective assessment based on the evidence. Be direct and clear.
State your verdict as one of: **CONFIRMED**, **STRONGLY SUPPORTED**, **LIKELY TRUE**, **INCONCLUSIVE**, **LIKELY FALSE**, **STRONGLY REFUTED**, **REFUTED**

Be thorough but concise. Cite sources. Your goal is truth, not diplomacy.

After the verdict narrative, emit EXACTLY ONE fenced ```json block as the final element of your response:

```json
{{
  "verdict": "<one of: CONFIRMED, STRONGLY SUPPORTED, LIKELY TRUE, INCONCLUSIVE, LIKELY FALSE, STRONGLY REFUTED, REFUTED>",
  "verdict_confidence": <0.0-1.0 float>,
  "rationale": "<one sentence>"
}}
```

`verdict_confidence` is YOUR confidence in the verdict. 0.9+ means very sure; ≈0.5 is a coin flip. Nothing after the closing fence."""


def build_decomposer_prompt(defender_text: str, skeptic_text: str,
                            max_per_side: int) -> str:
    """Stage A of the atomic fact-checker: extract atomic factual claims as JSON."""
    return f"""You are a CLAIM DECOMPOSER. Your job is to extract atomic factual claims from a debate round and emit them as a single JSON list.

DEFENDER'S ARGUMENT (Round response):
{defender_text[:6000]}

SKEPTIC'S ARGUMENT (Round response):
{skeptic_text[:6000]}

INSTRUCTIONS:
- Extract individual factual claims — each item must be one subject + one predicate (a single assertion that can be independently verified).
- "Atomic" examples: "The MMR vaccine is administered to children at age 1 in the US." / "Wakefield's 1998 study was retracted by The Lancet in 2010."
- NOT atomic: "Vaccines are safe and effective" (two claims, too broad). Skip rhetorical statements, opinions, conditionals ("if X then Y"), meta-arguments ("the skeptic failed to address X"), and value judgments.
- Cap at {max_per_side} most-load-bearing atomic claims PER SIDE. Pick the claims most central to each side's case.
- Assign stable claim_ids of the form "cl_def_001" (defender) / "cl_skp_001" (skeptic), incrementing per side starting at 001.
- "key_terms" is a short list of the most search-friendly phrases from the claim (proper nouns, specific numbers, dated events).

OUTPUT FORMAT — emit EXACTLY ONE fenced ```json block as the final element of your response. Do not output anything after the closing fence. The block content must be a JSON array:

```json
[
  {{"claim_id": "cl_def_001", "side": "defender", "text": "<atomic factual claim>", "key_terms": ["term1", "term2"]}},
  {{"claim_id": "cl_skp_001", "side": "skeptic",  "text": "<atomic factual claim>", "key_terms": ["term1"]}}
]
```

If a side has no extractable atomic claims, omit it from the list rather than padding."""


def build_verification_question_prompt(atomic_claim_text: str,
                                       max_questions: int) -> str:
    """Stage B of the atomic fact-checker: generate independent verification questions."""
    return f"""You are a VERIFICATION QUESTION GENERATOR. Given ONE atomic factual claim, output 1-{max_questions} questions whose answers — retrieved via independent web search — would confirm or refute the claim.

CRITICAL RULES (Chain-of-Verification methodology):
- Each question must be answerable by a search engine WITHOUT reference to any surrounding debate context. The search executor sees ONLY your question.
- You MUST NOT reference "the defender", "the skeptic", "the claim", "this debate", or any meta-context. Phrase questions as standalone factual queries.
- Aim to elicit the underlying ground truth, not to confirm a position. Bad: "Is XYZ really 100?" Good: "What is the actual value of XYZ?"
- If the claim has multiple verifiable components (e.g., "Study X by Author Y in 1998 found Z"), generate questions that target each component independently.
- Hard cap: emit at most {max_questions} questions.

ATOMIC CLAIM TO VERIFY:
"{atomic_claim_text}"

OUTPUT FORMAT — emit EXACTLY ONE fenced ```json block as the final element of your response. Nothing after the closing fence:

```json
["question 1", "question 2"]
```"""


def build_atomic_fact_check_prompt(atomic_claim_text: str,
                                   qa_pairs: list[dict]) -> str:
    """Stage C of the atomic fact-checker: per-claim verdict with structured output."""
    qa_block = (
        "\n\n".join(
            f"Q{i+1}: {qa['question']}\nA{i+1}:\n{qa['answer'][:2000]}"
            for i, qa in enumerate(qa_pairs)
        )
        if qa_pairs else "(No research conducted for this claim.)"
    )
    return f"""You are an ATOMIC FACT-CHECKER. Given ONE atomic claim and the independent search-derived evidence below, issue a single per-claim verdict as JSON.

ATOMIC CLAIM:
"{atomic_claim_text}"

INDEPENDENT EVIDENCE (each question was searched in isolation, with no shared context):
{qa_block}

VERDICT TAXONOMY (pick exactly one):
- VERIFIED — multiple independent credible sources confirm the claim
- LIKELY — supporting evidence found, minor gaps remain
- UNVERIFIED — insufficient evidence to confirm or deny
- DISPUTED — credible sources directly contradict each other
- FALSE — clearly disproven by credible evidence

SOURCE TIER (assign the STRONGEST tier of source you actually rely on):
- PRIMARY: peer-reviewed papers, official government statistics, court documents, regulatory filings
- SECONDARY: Reuters / AP / BBC / institutional press releases / established news desks
- TERTIARY: other news, explanatory journalism, Wikipedia
- LOW: blogs, social posts, user-generated content, undated sources, or no source available

GUIDANCE:
- Be DIRECT. Do not hedge with UNVERIFIED when the evidence actually points one way.
- Confidence in [0.0, 1.0]: 0.9+ for VERIFIED/FALSE with PRIMARY sources; 0.3-0.5 for UNVERIFIED; lower for ambiguous evidence.
- Evidence URL: the single most authoritative URL from the research above. If none, use "".
- Rationale: one sentence explaining the verdict in plain language.

OUTPUT FORMAT — emit EXACTLY ONE fenced ```json block as the final element of your response. Nothing after the closing fence:

```json
{{
  "verdict": "VERIFIED|LIKELY|UNVERIFIED|DISPUTED|FALSE",
  "confidence": 0.0,
  "evidence_url": "https://...",
  "source_tier": "PRIMARY|SECONDARY|TERTIARY|LOW",
  "rationale": "one sentence"
}}
```"""


def build_claim_classifier_prompt(claim: str) -> str:
    class_list = ", ".join(CLAIM_CLASSES)
    return f"""You are a CLAIM CLASSIFIER. Your job is to read a single claim and route it to one of four classes so the debate system can pick the right policy. Be decisive — pick exactly one class.

CLAIM: "{claim}"

CLASSES (pick exactly one):

- **FACTUAL_TESTABLE** — the claim asserts something about the past or present that can in principle be checked against evidence (a specific event happened, a specific number is correct, a product has a specific property, a person did a specific thing). Example: "Antoniu Gugu is a bad engineer" (testable via track record), "The microneedling PDRN treatment is FDA-approved" (testable via FDA records).
- **PREDICTIVE_FUTURE** — the claim is about a future event that has not yet occurred and cannot be checked now. Example: "The S&P 500 will be down significantly next week", "The next US president will be a Democrat".
- **VALUE_NORMATIVE** — the claim is about values, ethics, preferences, or aesthetic judgments where no empirical test settles it. Example: "Capitalism is unjust", "Pineapple does not belong on pizza".
- **HARM_INDUCING_FALSEHOOD** — the claim is a well-established, scientifically refuted falsehood that, if defended adversarially, risks amplifying real-world harm (anti-vaccine claims, denial of well-documented atrocities, conspiracy claims about disasters, etc.). Use this class ONLY for claims where the falsehood is established by overwhelming scientific or institutional consensus AND defending it could cause meaningful harm. When in doubt, prefer FACTUAL_TESTABLE.

GUIDANCE:
- Be conservative with HARM_INDUCING_FALSEHOOD. It is for claims like "vaccines cause autism", "the moon landing was faked", "the Holocaust did not happen". It is NOT for contested-but-debatable claims, fringe-but-non-harmful claims, or claims merely about a person's competence.
- PREDICTIVE_FUTURE takes precedence over FACTUAL_TESTABLE when the claim explicitly references future events that have not occurred.
- VALUE_NORMATIVE applies only when no empirical test could settle the claim. If the claim has any testable factual component, prefer FACTUAL_TESTABLE.

Respond in this EXACT format:
REASONING: one sentence explaining which class fits and why.
CLASS: <one of: {class_list}>

The CLASS line MUST be on its own line and contain exactly one of the four tokens above (uppercase, with underscore)."""


def build_researcher_prompt(query: str, claim: str, requesting_agent: str) -> str:
    return f"""Search the web for factual information about the following query. Provide a concise, evidence-based summary with source URLs where available.

This research was requested by the {requesting_agent} in a debate about: "{claim}"

Query: {query}

Provide factual findings only. No opinions. Include source URLs inline as markdown links.

SOURCE TIER TAGGING (required):
For each distinct finding you cite, append a tier tag immediately after the citation in the form `[TIER: PRIMARY|SECONDARY|TERTIARY|LOW]`. Use the strongest tier that genuinely supports the finding:
- PRIMARY: peer-reviewed papers, official government statistics (BLS, WHO, government registries), court documents, regulatory filings, primary clinical trial data.
- SECONDARY: Reuters / AP / BBC / institutional press releases / established news desks reporting first-hand.
- TERTIARY: other news, explanatory journalism, Wikipedia, secondary aggregators.
- LOW: blogs, social posts, user-generated content, undated sources, or anything you cannot trace to a credible original.

Example finding line: "FDA approved drug X in 2023 ([source](https://fda.gov/...)). [TIER: PRIMARY]"

If you cannot find a credible source, say so explicitly and tag [TIER: LOW]."""
