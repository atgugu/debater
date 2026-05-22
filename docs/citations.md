# Citations — Truth-Seeking & Debiasing Methods for `debater`

This bibliography supports the prompting audit and improvement plan in `/home/atgu/.claude/plans/analyse-prompting-do-research-twinkly-gadget.md`. Each entry was verified via web search in May 2026; arXiv IDs in the 26YY format follow the YYMM-submission scheme used by arXiv (e.g. 2604.xxxxx = April 2026 submission).

Entries are grouped by the topic they primarily inform. Where a paper is load-bearing for a specific recommendation, the section number from the plan is annotated in brackets, e.g. **[Plan §1]**.

---

## 1. Debate as Scalable Oversight

The intellectual lineage that motivates this whole project, plus the 2024–2026 work that revises the picture.

- **Irving, G., Christiano, P., & Amodei, D. (2018).** *AI Safety via Debate.* arXiv:1805.00899. https://arxiv.org/abs/1805.00899
  - The original proposal: two AI agents debate, a (human or weaker AI) judge decides. The seed for every multi-agent debate paper since.

- **Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023).** *Improving Factuality and Reasoning in Language Models through Multiagent Debate.* ICML 2024. arXiv:2305.14325. https://arxiv.org/abs/2305.14325
  - Empirical case that multiple LLM instances debating each other beats single-agent baselines on math + factual reasoning. The optimistic side of the debate-helps-truth question.

- **Khan, A., Hughes, J., Valentine, D., Ruis, L., Sachan, K., Radhakrishnan, A., Grefenstette, E., Bowman, S. R., Rocktäschel, T., & Perez, E. (2024).** *Debating with More Persuasive LLMs Leads to More Truthful Answers.* ICML 2024 (Best Paper). arXiv:2402.06782. https://arxiv.org/abs/2402.06782 **[Plan §5]**
  - Non-expert judges reach 76% (LLM) / 88% (human) accuracy on QuALITY when reading expert-LLM debates, vs 48% / 60% on naive baselines. Optimizing debaters for persuasiveness *unsupervisedly* improves the non-expert's ability to identify truth.

- **Smit, A. P., Grinsztajn, N., Duckworth, P., Barrett, T. D., & Pretorius, A. (2024).** *Should We Be Going MAD? A Look at Multi-Agent Debate Strategies for LLMs.* ICML 2024. arXiv:2311.17371. https://arxiv.org/abs/2311.17371 **[Plan §1, §2]**
  - Critical finding: multi-agent debate does **not** reliably outperform self-consistency or ensembling at fixed compute. The MAD-is-better intuition is fragile; gains require careful hyperparameter tuning and especially model heterogeneity.

- **Zhang, H., Cui, Z., Wang, X., Zhang, Q., Wang, Z., Wu, D., & Hu, S. (2025).** *If Multi-Agent Debate is the Answer, What is the Question?* (also titled in v2 *Stop Overvaluing Multi-Agent Debate — We Must Rethink Evaluation and Embrace Model Heterogeneity*). arXiv:2502.08788. https://arxiv.org/abs/2502.08788 **[Plan §1, §2]**
  - Systematic re-evaluation across 5 MAD methods × 9 benchmarks × 4 models. MAD beats CoT + self-consistency *only* when models are heterogeneous. The single sharpest argument for §1+§2.

- **Wu, H., Li, Z., & Li, L. (2025).** *Can LLM Agents Really Debate? A Controlled Study of Multi-Agent Debate in Logical Reasoning.* arXiv:2511.07784. https://arxiv.org/abs/2511.07784 **[Plan §1]**
  - Knight-Knave-Spy puzzle benchmark with verifiable ground truth. Conclusion: intrinsic reasoning strength and group diversity dominate; structural parameters like order or confidence visibility offer limited gains. Majority pressure suppresses independent correction.

- **He, H., Li, Y., Wen, D., Chen, Y., Cheng, R., Chen, D., & Lau, F. C. M. (2026).** *Debating Truth: Debate-driven Claim Verification with Multiple Large Language Model Agents.* WWW 2026. arXiv:2507.19090. https://arxiv.org/abs/2507.19090 **[Plan §5]**
  - DebateCV framework. Key insight that maps directly to our audit: **zero-shot moderators are biased toward neutral judgments** — exactly the INCONCLUSIVE-default failure mode this plan addresses. Their fix is Debate-SFT fine-tuning; ours is the deterministic verdict-policy guard.

---

## 2. LLM-as-a-Judge: Biases and Mitigations

The judge stage is where most truth-seeking debate systems fail. This is the most active sub-literature.

- **Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023).** *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023 D&B. arXiv:2306.05685. https://arxiv.org/abs/2306.05685
  - The canonical reference. Identifies position bias, verbosity bias, self-enhancement bias, and limited reasoning. Strong LLM judges (GPT-4) reach 80%+ agreement with humans, matching inter-human agreement.

- **Wang, P., Li, L., Chen, L., Cai, Z., Zhu, D., Lin, B., Cao, Y., Kong, L., Liu, Q., Liu, T., & Sui, Z. (2023).** *Large Language Models are not Fair Evaluators.* ACL 2024. arXiv:2305.17926. https://arxiv.org/abs/2305.17926 **[Plan §1]**
  - Shows quality rankings can be flipped by changing the order of candidates. Introduces three calibrations: Multiple Evidence Calibration, Balanced Position Calibration, Human-in-the-Loop. §1's order-swap aggregation is a direct application.

- **Li, R., Patel, T., & Du, X. (2023).** *PRD: Peer Rank and Discussion Improve Large Language Model based Evaluations.* arXiv:2307.02762. https://arxiv.org/abs/2307.02762 **[Plan §1, §7]**
  - Peer Rank algorithm + Peer Discussion: multiple LLMs each rate, then discuss to reach mutual agreement. Higher accuracy and better alignment with human judgment than single-LLM-as-judge.

- **Koo, R., Lee, M., Raheja, V., Park, J. I., Kim, Z. M., & Kang, D. (2023).** *Benchmarking Cognitive Biases in Large Language Models as Evaluators (CoBBLEr).* arXiv:2309.17012. https://arxiv.org/abs/2309.17012
  - Benchmarks six cognitive biases: order, compassion-fade, egocentric, bandwagon, attentional, verbosity. Average ~40% of LLM-judge comparisons show bias indication. Sets the empirical ceiling on how much prompt-based debiasing alone can do.

- **Panickssery, A., Bowman, S. R., & Feng, S. (2024).** *LLM Evaluators Recognize and Favor Their Own Generations.* NeurIPS 2024. arXiv:2404.13076. https://arxiv.org/abs/2404.13076 **[Plan §1, §2]**
  - GPT-4 and Llama-2 distinguish their own outputs from others' non-trivially and favor them. Self-recognition and self-preference are linearly correlated. **A panel of one model family is the worst-case configuration** — direct justification for heterogeneous jury.

- **Verga, P., Hofstatter, S., Althammer, S., Su, Y., Piktus, A., Arkhangorodsky, A., Xu, M., White, N., & Lewis, P. (2024).** *Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models (PoLL).* arXiv:2404.18796. https://arxiv.org/abs/2404.18796 **[Plan §1, §7]**
  - A 3-model panel (Command R + GPT-3.5 + Claude Haiku) correlates with human judgment better than GPT-4 alone, at ~1/7 the cost. The cost-effectiveness argument for §1.

- **Sharma, M., Tong, M., Korbak, T., Duvenaud, D., Askell, A., Bowman, S. R., et al. (2023).** *Towards Understanding Sycophancy in Language Models.* Anthropic. arXiv:2310.13548. https://arxiv.org/abs/2310.13548 **[Plan §6]**
  - Sycophancy is a general property of RLHF'd assistants — they prefer convincingly-written but false answers over plainly-stated truthful ones. Forcing models to "steel-man" falsehoods is a sycophancy-inducing setup; the pre-debate classifier in §6 routes around it.

- **Kim, S., Suk, J., Longpre, S., Lin, B. Y., Shin, J., Welleck, S., Neubig, G., Lee, M., Lee, K., & Seo, M. (2024).** *Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models.* EMNLP 2024. arXiv:2405.01535. https://arxiv.org/abs/2405.01535
  - Open-source 13B/8x7B fine-tuned evaluator that matches GPT-4 correlation with humans on customized rubrics. *Background* — not adopted (requires non-Anthropic infrastructure) but the canonical reference for fine-tuned-evaluator alternatives.

- **Soumik et al. (2026).** *Judging the Judges: A Systematic Evaluation of Bias Mitigation Strategies in LLM-as-a-Judge Pipelines.* arXiv:2604.23178. https://arxiv.org/abs/2604.23178 **[Plan §1, "Strong reversals"]**
  - 9 debiasing strategies × 5 judges × 3 benchmarks × 4 bias types. Two reversal findings: **(a) style bias (0.76-0.92) dominates position bias (≤0.04) by ~20×** — most attention has gone to the smaller bias; **(b) combined debiasing gives +11.2 pp on Claude Sonnet 4** (p<0.0001).

---

## 3. Factuality, Verification, and Hallucination Detection

This is the second-most-load-bearing area for `debater` — the fact-checker role.

- **Manakul, P., Liusie, A., & Gales, M. J. F. (2023).** *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models.* EMNLP 2023. arXiv:2303.08896. https://arxiv.org/abs/2303.08896 **[Plan §7]**
  - Resample N completions of the same prompt; inconsistency across samples ⇒ hallucination signal. Combines well with §7's self-consistency.

- **Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W.-T., Koh, P. W., Iyyer, M., Zettlemoyer, L., & Hajishirzi, H. (2023).** *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP 2023. arXiv:2305.14251. https://arxiv.org/abs/2305.14251 **[Plan §3]**
  - Decompose generation into atomic facts; compute % supported by a knowledge source. The "decompose before verify" pattern that §3 adopts.

- **Dhuliawala, S., Komeili, M., Xu, J., Raileanu, R., Li, X., Celikyilmaz, A., & Weston, J. (2023).** *Chain-of-Verification Reduces Hallucination in Large Language Models.* arXiv:2309.11495. https://arxiv.org/abs/2309.11495 **[Plan §3]**
  - CoVe: (1) draft, (2) plan verification questions, (3) answer each question **independently** (the crucial step — not conditioned on the draft), (4) revise. Consistent hallucination reductions across list-QA, MultiSpanQA, longform.

- **Wei, J., Yang, C., Song, X., Lu, Y., Hu, N., Tran, D., Peng, D., Liu, R., Huang, D., Du, C., & Le, Q. V. (2024).** *Long-form factuality in large language models (SAFE).* NeurIPS 2024. Google DeepMind. arXiv:2403.18802. https://arxiv.org/abs/2403.18802 **[Plan §3]**
  - SAFE: split a long response into atomic facts, send Google searches per fact, decide support with multi-step reasoning. 72% agreement with human annotators, wins 76% of disagreement cases, 20× cheaper than crowdsourcing. The structural template for §3.

- **Bohnet, B., Tran, V. Q., Verga, P., Aharoni, R., Andor, D., Soares, L. B., et al. (2022).** *Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models.* Google Research. arXiv:2212.08037. https://arxiv.org/abs/2212.08037 **[Plan §3]**
  - The original argument that attribution (citation-faithfulness) must be evaluated, not assumed. Foundational for treating defender/skeptic citations as evidence-to-verify rather than evidence-as-given.

- **Huang, Y., Ribeiro, L. F. R., Hardalov, M., Dhingra, B., Dreyer, M., & Saligrama, V. (2026).** *DeepFact: Co-Evolving Benchmarks and Agents for Deep Research Factuality.* arXiv:2603.05912. https://arxiv.org/abs/2603.05912 **[Plan §3]**
  - Document-level factuality verification with **auditable rationales** (Audit-then-Score). After 4 AtS rounds, expert micro-gold accuracy reaches 90.9%. Directly applicable to our multi-round debate transcripts.

- **Wan, Y., Fang, T., et al. (2026).** *Inference-Time Scaling of Verification: Self-Evolving Deep Research Agents via Test-Time Rubric-Guided Verification (DeepVerifier).* arXiv:2601.15808. https://arxiv.org/abs/2601.15808 **[Plan §2, §4]**
  - Rubric-based outcome verifier. Beats vanilla agent-as-judge and LLM-judge baselines by 12-48% on meta-eval F1; test-time verification yields 8-11% accuracy gains on GAIA subsets when paired with capable closed-source LLMs. Concrete evidence that extended-thinking-on-judging is worth the cost.

---

## 4. Reasoning, Calibration, and Test-Time Compute

How to get better answers per-question without retraining.

- **Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2022).** *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR 2023. arXiv:2203.11171. https://arxiv.org/abs/2203.11171 **[Plan §7]**
  - Sample N CoT paths, marginalize over reasoning, pick most-consistent answer. +5–18% on GSM8K, SVAMP, AQuA, StrategyQA, ARC-challenge. The minimum-viable version of test-time compute scaling.

- **Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., Schiefer, N., Hatfield-Dodds, Z., et al. (2022).** *Language Models (Mostly) Know What They Know.* Anthropic. arXiv:2207.05221. https://arxiv.org/abs/2207.05221 **[Plan §4]**
  - P(True) and P(IK) self-evaluation are well-calibrated at scale; the model "mostly knows" what it knows. Foundation for asking the evaluator/summarizer for verbalized confidence.

- **Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., Chen, A., Goldie, A., et al. (2022).** *Constitutional AI: Harmlessness from AI Feedback.* Anthropic. arXiv:2212.08073. https://arxiv.org/abs/2212.08073
  - RLAIF: self-critique + revise loop trained against a constitution. The intellectual basis for any system that asks an LLM to score another LLM's output against a written rubric — which is everything in `debater`.

- **Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., et al. (2023).** *Self-Refine: Iterative Refinement with Self-Feedback.* NeurIPS 2023. arXiv:2303.17651. https://arxiv.org/abs/2303.17651
  - Single LLM, three roles (generator, critic, refiner), iterative loop. ~20% improvement across 7 tasks. Caveat: see Huang 2024 for the negative result on intrinsic self-correction.

- **Tian, K., Mitchell, E., Zhou, A., Sharma, A., Rafailov, R., Yao, H., Finn, C., & Manning, C. D. (2023).** *Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback.* EMNLP 2023. arXiv:2305.14975. https://arxiv.org/abs/2305.14975 **[Plan §4]**
  - For RLHF'd models (ChatGPT, GPT-4, Claude), **verbalized confidence is better calibrated than token-probability confidence**, often halving expected calibration error (TriviaQA, SciQ, TruthfulQA). Direct prescription for §4.

- **Huang, J., Chen, X., Mishra, S., Zheng, H. S., Yu, A. W., Song, X., & Zhou, D. (2024).** *Large Language Models Cannot Self-Correct Reasoning Yet.* ICLR 2024. arXiv:2310.01798. https://arxiv.org/abs/2310.01798
  - **Negative result** that constrains the design space: intrinsic self-correction (no external feedback) often makes outputs *worse*. Justifies why §3 uses external search-based verification rather than re-prompting the same model.

- **Snell, C., Lee, J., Xu, K., & Kumar, A. (2024).** *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* arXiv:2408.03314. https://arxiv.org/abs/2408.03314 **[Plan §2]**
  - Compute-optimal inference can beat a 14× larger base model on solvable problems; 4× improvement over best-of-N. The strongest empirical case for spending more on judgment-time compute (extended thinking, jury) rather than always upgrading the base model.

---

## 5. Background (referenced but not adopted in this iteration)

Useful intellectual context, but out-of-scope for a CLI-wrapper system:

- **Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., & Narasimhan, K. (2023).** *Tree of Thoughts: Deliberate Problem Solving with Large Language Models.* NeurIPS 2023. arXiv:2305.10601. https://arxiv.org/abs/2305.10601
  - Branching exploration over reasoning steps. Powerful for single-turn problem-solving (74% on Game of 24 vs 4% for GPT-4 CoT). Adversarial debate already provides much of the branching value.

- **Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., & Cobbe, K. (2023).** *Let's Verify Step by Step.* OpenAI. arXiv:2305.20050. https://arxiv.org/abs/2305.20050
  - Process supervision (per-step labels) beats outcome supervision for math. Training-time; relevant if we were building a custom verifier.

- **Kumar, A., Zhuang, V., Agarwal, R., Su, Y., et al. (2024).** *Training Language Models to Self-Correct via Reinforcement Learning (SCoRe).* Google DeepMind. arXiv:2409.12917. https://arxiv.org/abs/2409.12917
  - Multi-turn RL approach to train self-correction without external supervision. Training-time; not applicable to a CLI wrapper around Claude.

- **DeepSeek-AI (2025).** *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* arXiv:2501.12948. https://arxiv.org/abs/2501.12948
  - The RLVR / reasoning-model breakthrough — relevant as background to "why extended-thinking on the judge matters" but not directly applied.

---

## Maintenance notes

- All arXiv links are stable; URLs were verified via WebSearch in May 2026.
- 2026 arXiv IDs (`26YY.xxxxx` where `YY ≥ 01`) follow arXiv's YYMM submission scheme. They are real and resolve as expected.
- When citing this file in code comments or docs, prefer the form: `arXiv:2402.06782` (Khan et al., 2024). The arXiv ID is the durable identifier; titles can change between versions.
- Re-verify the 2026 references annually — some may move to peer-reviewed venues with revised titles.
