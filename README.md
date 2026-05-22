# debater

> Adversarial multi-agent debate for truth-seeking. Eight specialized Claude agents argue both sides of a claim, fact-check each other against the web, and produce a calibrated verdict — with explicit mitigations for LLM-as-judge biases.

`debater` runs N rounds of structured debate between a **Defender** and a **Skeptic**, with an atomic-claim **Fact-checker** independently verifying every load-bearing assertion, an **Evaluator** scoring each round on an absolute 0–10 rubric, and a **Summarizer** issuing a final verdict from `CONFIRMED` to `REFUTED`. A **Researcher** runs targeted web searches on demand. An **Analyst** handles claims the adversarial format shouldn't tackle (e.g. well-established harmful falsehoods).

Each debate is persisted on disk as `state.json` + per-role markdown logs + JSON sidecars. A web UI streams round-by-round progress and accepts mid-debate interjections.

```bash
debater                 # http://localhost:8081, then open in browser
```

## Agents

| Agent | Role |
|---|---|
| **Claim classifier** | Pre-debate routing: `FACTUAL_TESTABLE` / `PREDICTIVE_FUTURE` / `VALUE_NORMATIVE` / `HARM_INDUCING_FALSEHOOD` |
| **Defender** | Steel-mans the claim with the strongest evidence-backed argument |
| **Skeptic** | Critically challenges the claim and the defender's reasoning |
| **Fact-checker** | Atomic decomposition + per-claim Chain-of-Verification via independent search |
| **Evaluator** | Scores each round on an absolute 0–10 rubric, N=3 samples with median aggregation |
| **Summarizer** | Final verdict + verbalized confidence |
| **Researcher** | On-demand web search with source-tier tagging |
| **Analyst** | Single-agent direct analysis for `HARM_INDUCING_FALSEHOOD` claims and refusal fallbacks |

All eight roles default to `claude-haiku-4-5`. Per-role models are overridable via `state.settings.models`.

## Truth-seeking techniques

Methodology drawn from the 2022–2026 literature on factuality, LLM-as-judge bias, and scalable oversight. Full bibliography in [`docs/citations.md`](docs/citations.md).

### Atomic fact-check pipeline (SAFE + CoVe)

Single-pass fact-checkers miss things. The fact-checker is a three-stage pipeline:

1. **Decompose** defender + skeptic responses into atomic factual claims (one subject + one predicate each)
2. **Generate verification questions** per atomic claim — phrased *without* any reference to the original debate context (Chain-of-Verification's central insight, which prevents bias bleed-through from the original response)
3. **Search each question independently** and emit a per-claim verdict (`VERIFIED` / `LIKELY` / `UNVERIFIED` / `DISPUTED` / `FALSE`) with confidence and evidence URL

Results are persisted as `round_N/fact_checker.json` sidecars for downstream tooling. Atomic decomposition with independent verification achieves ~72% agreement with crowd annotators at ~20× lower cost than holistic fact-checking [^safe][^cove][^factscore].

### Verdict-policy guard against neutral-bias

Zero-shot moderator LLMs have a well-documented bias toward `INCONCLUSIVE` even when evidence is directional. After the summarizer runs, a **deterministic post-hoc guard** overrides `INCONCLUSIVE → LIKELY TRUE/FALSE` when **all** of these hold:

- claim is `FACTUAL_TESTABLE`,
- a majority of rounds had a per-side score gap ≥ 2 points,
- the trailing side accumulated ≥ 3 `FALSE`/`DISPUTED` atomic claims,
- the leading side accumulated ≤ 1.

For predictive-future and value-normative claims the guard is intentionally inert — `INCONCLUSIVE` is the honest verdict there [^moderator-bias].

### Verbalized confidence + structured output

The evaluator, summarizer, and analyst all emit a fenced ` ```json ` block alongside their narrative containing scores, verdicts, and **verbalized confidence** in `[0.0, 1.0]`. Verbalized confidence is significantly better-calibrated than logit probability for RLHF'd models like Claude — roughly 50% relative ECE reduction [^calibration]. The JSON parser is tried first; regex / LLM-extractor fallbacks preserve backward compatibility with older debates.

### Evaluator self-consistency

The evaluator runs `N=3` samples per round (configurable via `evaluator_samples`). Per-side scores are aggregated by **median**, per-side confidence by **mean**. Sample-level spread > 3 raises an `evaluator_unstable` flag on the round. Cheap variance reduction that approximates the gain of a multi-judge jury without the cost [^self-consistency].

### Pre-debate routing avoids sycophancy

Every claim is classified before the debate starts. `HARM_INDUCING_FALSEHOOD` claims — well-established refuted falsehoods whose adversarial defense would amplify misinformation (anti-vaccine claims, Holocaust denial, etc.) — skip the debate entirely and route to the analyst with appropriate framing. Forcing models to argue convincingly for falsehoods makes them sycophantic, not truthful [^sycophancy].

### Explicit prompt-level debiasing

The evaluator's rubric addresses five well-replicated LLM-as-judge biases at scoring time:

- **POSITION** — defender is shown first by convention; evaluator told to score independently [^position-bias]
- **VERBOSITY** — argument quality ≠ argument length [^verbosity-bias]
- **CONFIDENCE** — bold delivery ≠ strong evidence
- **RHETORIC** — emotional appeal / metaphors / authority tone are not evidence
- **INDEPENDENCE** — score this round on its own merits; ignore cumulative momentum

Plus a **non-symmetric absolute 0–10 rubric**: scores need not sum to any total. Both sides can score low when both are weak; both can score high when both are excellent.

### Source-tier weighting

Each finding the researcher returns is tagged `[TIER: PRIMARY|SECONDARY|TERTIARY|LOW]` (peer-reviewed papers / established news desks / explanatory journalism / blogs respectively). The fact-checker's per-claim verdict carries the strongest tier of source it actually relied on. Tier signal is persisted alongside each atomic claim for downstream weighting.

### Robust refusal handling

If defender or skeptic refuses to engage on a sensitive claim, the orchestrator retries once with a research-context nudge. A second refusal transitions cleanly to single-agent analyst mode (`status: analyzing → complete`), persisting `analysis.md` + verdict. Analyst failures are explicitly recorded as `status: analysis_failed` rather than silently halting.

## Install

```bash
git clone https://github.com/yourname/debater.git
cd debater
pip install -e ".[dev]"
```

Requires Python 3.10+ and the [Claude CLI](https://docs.claude.com/claude-code) on `PATH`.

## Run

```bash
debater                              # http://localhost:8081
debater --port 9000                  # custom port
debater --max-concurrent-claude 4    # more CLI parallelism
debater --debates-dir /tmp/regress   # alternative state dir
```

### CLI flags

| Flag | Default | Notes |
|---|---|---|
| `--port` | `8081` | HTTP port |
| `--debates-dir` | `debates` | Where state lives on disk |
| `--max-concurrent-claude` | `2` | Cap on concurrent Claude CLI subprocesses |

### Per-debate settings

These knobs live in `state.settings` per debate and are overridable via the API:

| Setting | Default | Purpose |
|---|---|---|
| `max_iterations` | `5` | Maximum debate rounds |
| `early_stop_threshold` | `3` | Halt when scores stabilize for N consecutive rounds |
| `models` | all `claude-haiku-4-5` | Per-role model override |
| `efforts` | all `null` | Per-role extended-thinking effort |
| `max_atomic_claims_per_side` | `4` | Atomic fact-check cap per round per side |
| `max_questions_per_claim` | `2` | Verification questions per atomic claim |
| `evaluator_samples` | `3` | Evaluator self-consistency sample count |

## Project layout

```
src/debater/
  cli.py            entry point + signal handling
  config.py         constants (models, verdict scale, claim classes)
  registry.py       runtime state owner (locks, threads, subprocesses)
  state.py          state.json I/O, debate directory layout
  claude_runner.py  Claude CLI subprocess wrapper
  prompts.py        all agent prompt templates
  research.py       research-request handling + per-question runner
  orchestrator.py   round cycle, atomic FC, self-consistency, verdict guard
  server.py         HTTP routes + handler factory
docs/citations.md   bibliography of techniques
static/index.html   single-page UI (Tailwind + vanilla JS)
tests/              pytest suite — 212 tests, Claude CLI fully mocked
debates/            persisted debate state (one directory per debate)
```

## Development

```bash
pip install -e ".[dev]"
pytest                     # 212 tests, runs in ~16s
ruff check src tests       # lint
ruff format src tests      # auto-format
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for module layout notes and the lock-nesting invariant.

## Roadmap

Deferred from the current Haiku-only iteration:

- **Heterogeneous evaluator jury** — Haiku + Sonnet + Opus mix to break cross-model self-preference [^self-preference]
- **Stronger models on keystone roles** — fact-checker / summarizer / evaluator on Sonnet 4.6 or Opus 4.7 with extended thinking [^test-time-compute]
- **Order-swap on evaluator self-consistency** — "Participant A/B" relabeling on samples 2–3 to mitigate position bias structurally
- **Parallel atomic FC pipeline** — `ThreadPoolExecutor` over the per-claim outer loop (the `claude_semaphore` already throttles)

## License

MIT — see [LICENSE](LICENSE).

---

### References

[^safe]: Wei et al. 2024, *Long-form factuality in large language models (SAFE)*, NeurIPS. [arXiv:2403.18802](https://arxiv.org/abs/2403.18802)
[^cove]: Dhuliawala et al. 2023, *Chain-of-Verification Reduces Hallucination in Large Language Models*. [arXiv:2309.11495](https://arxiv.org/abs/2309.11495)
[^factscore]: Min et al. 2023, *FActScore: Fine-grained Atomic Evaluation of Factual Precision*, EMNLP. [arXiv:2305.14251](https://arxiv.org/abs/2305.14251)
[^moderator-bias]: He et al. 2026, *Debating Truth: Debate-driven Claim Verification with Multiple LLM Agents*, WWW 2026. [arXiv:2507.19090](https://arxiv.org/abs/2507.19090)
[^calibration]: Tian et al. 2023, *Just Ask for Calibration*, EMNLP. [arXiv:2305.14975](https://arxiv.org/abs/2305.14975)
[^self-consistency]: Wang et al. 2022, *Self-Consistency Improves Chain of Thought Reasoning*, ICLR 2023. [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
[^sycophancy]: Sharma et al. 2023, *Towards Understanding Sycophancy in Language Models* (Anthropic). [arXiv:2310.13548](https://arxiv.org/abs/2310.13548)
[^position-bias]: Wang et al. 2023, *Large Language Models are not Fair Evaluators*, ACL 2024. [arXiv:2305.17926](https://arxiv.org/abs/2305.17926)
[^verbosity-bias]: Zheng et al. 2023, *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, NeurIPS. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
[^self-preference]: Panickssery, Bowman & Feng 2024, *LLM Evaluators Recognize and Favor Their Own Generations*, NeurIPS. [arXiv:2404.13076](https://arxiv.org/abs/2404.13076)
[^test-time-compute]: Snell et al. 2024, *Scaling LLM Test-Time Compute Optimally*. [arXiv:2408.03314](https://arxiv.org/abs/2408.03314)
