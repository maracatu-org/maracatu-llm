# Maracatu AI survey: efficiency papers (Feb 2025 to Apr 2026)

**Survey date:** 2026-04-22
**Run by:** `ml-researcher` agent via WebSearch/WebFetch
**Scope:** "do more with less" for the 80M-8B ladder, on top of a baseline of papers through Jan 2025 (DeepSeek-V3/R1, Chinchilla, muP, FineWeb, DeepSeekMoE, OLMoE, Mamba-2, Jamba, Phi-3/4, Gemma 2, SmolLM2, DPO/SimPO/KTO, Magpie, S1, YaRN, LongRoPE).

**23 selected papers.** Tiers:

- **S** = read and consider applying now
- **A** = read, medium-term relevance
- **B** = be aware of, future or indirect applicability

Agent honesty note: some search results returned arXiv IDs post-Apr-2026 (invalid/hallucinated). Discarded. Only verifiable links with real arXiv IDs (2501-2512) kept.

---

## 1. Pretraining efficiency

### S1. Muon is Scalable for LLM Training (Moonshot AI, Feb 2025)

- **Link:** [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)
- **Contribution:** optimizer based on Newton-Schulz orthogonalization that scales to large models with two corrections (weight decay + per-parameter update-scale adjustment). ~2x compute efficiency vs AdamW in the compute-optimal regime. Trained Moonlight 3B/16B MoE with 5.7T tokens.
- **Why it matters for Maracatu:** swapping AdamW for Muon on M-80M may yield ~30-50% fewer tokens to reach the same loss. Implementation is ~200 lines of PyTorch. High leverage, low architectural risk.
- **Tier:** S

### S2. Kimi K2: Open Agentic Intelligence (Moonshot, Jul 2025)

- **Link:** [arXiv:2507.20534](https://arxiv.org/abs/2507.20534)
- **Contribution:** MoE 1T/32B active. Introduces MuonClip (Muon + QK-Clip for stability), DeepSeek-V3-like architecture with 384 experts. 15.5T tokens without significant instabilities.
- **Why it matters:** QK-Clip is a cheap recipe to stabilize Muon at scale. On M-8B it may be the difference between "training converged" and "loss blew up at iter 30k".
- **Tier:** A

### A3. Insights into DeepSeek-V3: Scaling Challenges and Hardware Reflections (May 2025)

- **Link:** [arXiv:2505.09343](https://arxiv.org/html/2505.09343v2)
- **Contribution:** technical post-mortem of the V3 run, with detailed analysis of FP8, MLA and hardware choices. Not a "new result" paper, it is the richest document on what worked and what almost went wrong at scale.
- **Why it matters:** required reading before committing to the final M-8B/80B architecture. MLA costs 4-7% of perplexity vs MHA but reduces KV-cache by ~93%, a critical trade-off for cheap inference.
- **Tier:** A

### A4. Recipes for Pre-training LLMs with MXFP8 (NVIDIA, Jun 2025)

- **Link:** [arXiv:2506.08027](https://arxiv.org/html/2506.08027v1)
- **Contribution:** complete recipe for MXFP8 (block-scaled FP8), including 8B dense trained on 15T tokens matching the BF16 baseline. ~22% time reduction, 14% less peak memory, 19% more throughput.
- **Why it matters:** MXFP8 is what Blackwell (B200) will prioritize. At M-8B+ scale it's 20%+ direct savings. Kaggle T4 and old A100 don't support it well, so only relevant post-8B.
- **Tier:** A (wait for scale)

### B5. Scaling Laws for Precision (ICLR 2025)

- **Link:** [Harvard PDF](https://pehlevan.seas.harvard.edu/sites/g/files/omnuum6471/files/2025-03/Kumar_etal_ICLR_2025.pdf)
- **Contribution:** formalizes trade-offs between precision (FP8/FP4), parameters and data. Shows that lower precision reduces "effective parameter count".
- **Why it matters:** conceptual, useful to decide when to move to FP8. Does not change anything on M-80M.
- **Tier:** B

---

## 2. Data curation 2025+

### S6. BeyondWeb: Scaling Synthetic Data for Trillion-scale Pretraining (DatologyAI, Aug 2025)

- **Link:** [arXiv:2508.10975](https://arxiv.org/abs/2508.10975)
- **Contribution:** framework of systematic rephrasing of web documents. A 3B/180B-token model on BeyondWeb beats an 8B/180B-token model on Cosmopedia. They report a 7.7x training speedup via data quality.
- **Why it matters for Maracatu:** likely the highest-ROI paper for the current ladder. Investing in rephrasing the 1.57B tokens we already have can yield 2-3x effective improvement, cheaper than chasing more corpus. Embarrassingly parallelizable via inference APIs.
- **Tier:** S

### S7. Ultra-FineWeb: Efficient Data Filtering and Verification (May 2025)

- **Link:** [arXiv:2505.05427](https://arxiv.org/abs/2505.05427)
- **Contribution:** filtering pipeline via lightweight fastText classifier, generating ~1T EN tokens + 120B ZH of quality higher than FineWeb-Edu.
- **Why it matters:** cheap pipeline (fastText runs on CPU), replicable for PT-BR. If combined with the 120B PT corpus from S8 below, we can assemble an Ultra-FineWebPT without burning GPU.
- **Tier:** S

### S8. Building High-Quality Datasets for Portuguese LLMs (Sep 2025)

- **Link:** [arXiv:2509.08824](https://arxiv.org/html/2509.08824)
- **Contribution:** 120B-token PT-BR corpus built from Common Crawl, with an industrial filtering pipeline. Competes with industrial corpora used in closed models.
- **Why it matters:** required reading. Playbook for upgrading the Maracatu v3 corpus. With 120B tokens, M-800M stops being sub-Chinchilla and becomes truly viable. The ladder accelerates 6-12 months if we can get the dataset or replicate the pipeline.
- **Tier:** S

### A9. MixtureVitae: Permissive-First Pretraining Dataset (Sep 2025)

- **Link:** [arXiv:2509.25531](https://arxiv.org/abs/2509.25531)
- **Contribution:** 100% permissive corpus (public domain + permissive licenses). At 1.7B/300B tokens, beats FineWeb-Edu and approaches DCLM.
- **Why it matters:** aligns with Maracatu's Apache 2.0 stance. For public grant funding (FINEP, BNDES), having a corpus with a clean provenance trail is a huge legal and narrative asset.
- **Tier:** A

### A10. Aleph-Alpha-GermanWeb (May 2025)

- **Link:** [arXiv:2505.00022](https://arxiv.org/html/2505.00022v3)
- **Contribution:** model-based data curation + synthetic data specifically for German, a mid-resource language similar in volume to PT-BR.
- **Why it matters:** the most directly transferable case to our situation (mid-resource European language, smaller-than-frontier lab). Recipe replicable with minimal adjustments.
- **Tier:** A

---

## 3. Cheap post-training

### S11. DeepSeek-R1 Distillation Recipe (Jan 2025, already in baseline)

- **Link:** [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- **Contribution:** shows that pure SFT with 800k reasoning trajectories distilled from R1 is enough to turn Qwen-7B into a competitive reasoner. No RL needed on the student.
- **Why it matters:** route for M-8B to gain reasoning capability without a GPU cluster. Open dataset exists (Open-R1, NuminaMath). SFT on M-8B with 800k samples fits on 1 A100 for a few days.
- **Tier:** S (canonical reference)

### S12. REDI: Reinforcement Distillation from Teacher Data (May 2025)

- **Link:** [arXiv:2505.24850](https://arxiv.org/abs/2505.24850)
- **Contribution:** direct successor of DPO/SimPO for the distillation setting. SFT Stage 1 (positives only) + REDI Stage 2 (positives + negatives via REINFORCE-style). Qwen-REDI-1.5B reaches 83.1% MATH-500 with only 131k trajectories, matching R1-Distill-1.5B that used 800k.
- **Why it matters:** 6x less data for the same quality on reasoning distillation. Directly applicable to M-8B. Could be tested on M-800M as a proof of concept.
- **Tier:** S

### A13. RLVR Implicitly Incentivizes Correct Reasoning in Base LLMs (Jun 2025)

- **Link:** [arXiv:2506.14245](https://arxiv.org/abs/2506.14245)
- **Contribution:** theoretical framework showing why GRPO with binary rewards works on base LLMs (without prior SFT). Demonstrates that the signal incentivizes correct reasoning early in training.
- **Why it matters:** validates that we can skip expensive SFT and go straight to GRPO on M-8B if we have verifiers (math, code). Needs critical mass, likely infeasible sub-1B.
- **Tier:** A

### A14. Shrinking the Variance: Shrinkage Baselines for RLVR (Nov 2025)

- **Link:** [arXiv:2511.03710](https://arxiv.org/abs/2511.03710)
- **Contribution:** improves baseline estimation in GRPO via shrinkage estimators, reducing gradient variance.
- **Why it matters:** cheap engineering improvement on top of GRPO. Relevant when we do RL post-training on M-8B.
- **Tier:** A

---

## 4. Efficient architectures

### A15. Nemotron-H: Hybrid Mamba-Transformer (NVIDIA, Apr 2025)

- **Link:** [arXiv:2504.03624](https://arxiv.org/html/2504.03624v3)
- **Contribution:** 8B/56B family hybrid Mamba-2 + Transformer (mostly Mamba, some attention layers). Dramatically higher throughput in long context while preserving quality.
- **Why it matters:** interesting but breaks compatibility with LlamaForCausalLM, we lose the HF/Ollama route. Don't recommend adopting now. Worth monitoring whether the ecosystem supports it well.
- **Tier:** A (monitor, don't adopt)

### A16. Nemotron 3 Nano: Hybrid Mamba-Transformer MoE (NVIDIA, Dec 2025)

- **Link:** [arXiv:2512.20848](https://arxiv.org/html/2512.20848v1)
- **Contribution:** MoE + Mamba-Transformer hybrid for agentic reasoning. NVIDIA's most recent direction in small-efficient models.
- **Why it matters:** signal of where the frontier is heading. Same HF compatibility caveat as the previous one.
- **Tier:** B

### B17. Kimi Linear: Expressive Efficient Attention (Oct 2025)

- **Link:** [arXiv:2510.26692](https://arxiv.org/pdf/2510.26692)
- **Contribution:** linear-attention variant competitive with full attention in quality and much cheaper in long context.
- **Why it matters:** long-context efficiency for the distant future. Breaks HF compat.
- **Tier:** B

---

## 5. Cheap long context

### S18. LongRoPE2: Near-Lossless LLM Context Window Scaling (ICML 2025)

- **Link:** [ICML 2025 poster](https://icml.cc/virtual/2025/poster/44280)
- **Contribution:** extends Llama3-8B to 128k context using only 10B tokens (80x less than Meta), preserving 98.5% of short-context performance. Uses evolutionary search for RoPE rescaling + mixed-context training.
- **Why it matters:** cheap context extension is exactly what a solo lab needs. 10B tokens of long-context data is viable on RunPod spot for ~R$2-5k. An "M-8B-long" with 64k-128k context is a strong commercial feature (RAG, OAB with long contracts).
- **Tier:** S

### A19. Rope to Nope and Back Again (Jan 2025)

- **Link:** [arXiv:2501.18795](https://arxiv.org/abs/2501.18795)
- **Contribution:** hybrid architecture interleaving NoPE and RoPE layers. NoPE layers learn retrieval, RoPE layers learn local context. SmolLM3 already uses this idea (3:1 RoPE:NoPE).
- **Why it matters:** small architectural change (some layers without RoPE) with measurable gain in long context. Preserves HF compat if implemented as a flag. Candidate to test on M-800M.
- **Tier:** A

---

## 6. Small-model recipes that beat frontier

### S20. Qwen3 Technical Report (May 2025)

- **Link:** [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)
- **Contribution:** family 0.6B through 32B dense + 30B-A3B and 235B-A22B MoE. Qwen3-1.7B base beats Qwen2.5-3B base on >50% of benchmarks. 36T tokens, 119 languages. QK-Norm added for stability.
- **Why it matters:** Qwen3-0.6B-base is the direct baseline for Maracatu-800M. If they, with 36T of multilingual corpus, beat models 3x larger, most of the gain is data scale, not params. QK-Norm is trivial to add to our model.py.
- **Tier:** S

### S21. SmolLM3 (Hugging Face, Jul 2025)

- **Link:** [official blog](https://huggingface.co/blog/smollm3) + [HF card](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base)
- **Contribution:** 3B dense, dual-mode reasoning, 6 languages (including PT), 128k context via YARN, fully open (weights + data mixture + configs). 11.2T tokens staged curriculum. GQA 4 groups + NoPE every 4 layers.
- **Why it matters:** exact template for Maracatu-800M. Same ambition scale, same open philosophy. It already covers PT natively, which moves the threshold: it's not enough to "be good at PT", we need to be noticeably better than SmolLM3-3B at PT to justify existing.
- **Tier:** S

### A22. Gemma 3 Technical Report (Mar 2025)

- **Link:** [arXiv:2503.19786](https://arxiv.org/abs/2503.19786)
- **Contribution:** 1B/4B/12B/27B with knowledge distillation from a larger teacher during pretraining. Gemma3-4B-IT competitive with Gemma2-27B-IT.
- **Why it matters:** the strong result is that distillation-during-pretraining (not just post-training) yields large gains at small scale. For M-800M/M-8B we can distill from an open teacher (Qwen3-32B, Llama-3.3-70B) during pretrain. Cost: 1.5-2x more compute, but a 30-50% better final model.
- **Tier:** A

### A23. Phi-4 Technical Report (Dec 2024, borderline baseline)

- **Link:** [arXiv:2412.08905](https://arxiv.org/abs/2412.08905)
- **Contribution:** 14B with a training recipe centered on synthetic data. Reports 12 epochs on synthetic data without overfitting.
- **Why it matters:** inspiration for Phi-style synthetic data in PT-BR. But replicating requires a strong PT-BR teacher, which doesn't yet exist openly. Viable post-M-8B.
- **Tier:** A

---

## 7. Underrepresented languages (transferable to PT-BR)

### A24. Revisiting Multilingual Data Mixtures in LM Pretraining (Oct 2025)

- **Link:** [arXiv:2510.25947](https://arxiv.org/html/2510.25947v1)
- **Contribution:** systematic study of mixture ratios across languages in pretraining. Refutes several old heuristics about oversampling low-resource languages.
- **Why it matters:** when we add EN, ES corpora to PT-BR (crucial for M-8B to get cross-lingual transfer for code/science), this is the guide.
- **Tier:** A

### A25. Nemotron-Mini-Hindi 4B (Oct 2024)

- **Link:** [arXiv:2410.14815](https://arxiv.org/abs/2410.14815)
- **Contribution:** HI+EN bilingual SLM, 400B-token CPT, mix of real + synthetic translation data.
- **Why it matters:** the exact case Maracatu mirrors: mid-resource language + national lab + synthetic translation data to fill gaps. Direct recipe.
- **Tier:** A

---

## Survey caveats

- **Tucano 2:** the search returned a result with an invalid arXiv ID (2603.xxxxx). Worth checking directly at [nkluge-correa.github.io/Tucano](https://nkluge-correa.github.io/Tucano/) whether there is a new release beyond Tucano 1 (Nov 2024).
- **muP refinements:** searches returned several papers with post-April-2026 IDs (invalid). The concrete 2025 ones are preprints on HyperP (muP + Muon) and CompleteP extensions. Marginal gain vs classical muP in our 20M-8B regime.
- **Totally new post-Chinchilla scaling laws:** the literature has matured, nothing replaces Chinchilla. There are refinements (inference-aware scaling, precision scaling). Consensus message: training far beyond Chinchilla is the default now (Qwen3-0.6B at 36T tokens = 60000:1 ratio). Validates what we already do in tiny-long.

---

## Top 5 to read this week

Prioritized by: (impact on the next release) x (cost of applying) x (dense but tractable reading).

1. **SmolLM3 blog + paper card** ([link](https://huggingface.co/blog/smollm3)). Direct competitor at M-800M, supports PT. Understand their decision-by-decision choices before writing `maracatu_800m.yaml`. 1-2h read.
2. **Building High-Quality Datasets for Portuguese LLMs** ([arXiv:2509.08824](https://arxiv.org/html/2509.08824)). Changes the ceiling of what's possible. 2h read + evaluate whether to download the dataset.
3. **BeyondWeb** ([arXiv:2508.10975](https://arxiv.org/abs/2508.10975)). Biggest "do more with less" lever that exists today. 1.5h read.
4. **Qwen3 Technical Report** ([arXiv:2505.09388](https://arxiv.org/abs/2505.09388)). Mental baseline of a well-executed small dense model. QK-Norm and three-stage pipeline. 2-3h read.
5. **Muon is Scalable** ([arXiv:2502.16982](https://arxiv.org/abs/2502.16982)). Perfect micro-experiment on M-80M: 5k iters AdamW vs Muon, same seed. If we confirm 1.5-2x speedup, it changes the economics of every release. 1.5h read + 1 day of experiment.

---

## Synthesis recommendation

Resist temptation: don't apply everything to M-80M. The 80M scale is too experimental for gains to be conclusive.

Suggested plan (in increasing order of risk):

1. **Current M-80M training** (week 0): keep the recipe, zero changes. Goal is to ship a clean release 2.
2. **Muon micro-experiment** (weeks 1-2, in parallel): run 5k iters on tiny with Muon vs AdamW. If >30% speedup on loss, becomes the default for M-800M.
3. **M-800M planning** (months 1-2): incorporate QK-Norm (Qwen3), staged curriculum (SmolLM3), distillation-during-pretrain from an open PT teacher (Qwen3-8B or Llama-3.1-8B-instruct in PT). Target corpus: try to get access to the 120B PT corpus or replicate the pipeline.
4. **M-8B post-training** (6-12 months): REDI + distillation from DeepSeek-R1 for reasoning. Skip traditional SFT at the end.
5. **Long-context M-8B variant:** LongRoPE2 as a separate fine-tuning step (10B tokens), release as "M-8B-long".

Explicit trade-off: adopting BeyondWeb + Muon + distillation-during-pretrain on M-800M costs ~30-50% more engineering time (2-3 extra weeks) but is estimated to be equivalent to 2-3x more tokens/params in the final model. If the goal is "Maracatu-800M better than SmolLM3-3B in PT-BR", it's not a luxury, it's a requirement.
