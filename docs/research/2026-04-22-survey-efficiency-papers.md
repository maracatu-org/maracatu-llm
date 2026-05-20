# Survey Maracatu AI: Papers de eficiência (fev/2025 a abr/2026)

**Data do survey:** 2026-04-22
**Executor:** agente `ml-researcher` via WebSearch/WebFetch
**Escopo:** "fazer mais com menos" pra ladder 80M-8B, em cima de baseline de papers até jan/2025 (DeepSeek-V3/R1, Chinchilla, muP, FineWeb, DeepSeekMoE, OLMoE, Mamba-2, Jamba, Phi-3/4, Gemma 2, SmolLM2, DPO/SimPO/KTO, Magpie, S1, YaRN, LongRoPE).

**23 papers selecionados.** Tiers:

- **S** = ler e considerar aplicar agora
- **A** = ler, relevância médio prazo
- **B** = conhecer, aplicabilidade futura ou indireta

Nota de honestidade do agente: alguns resultados retornados pelas buscas tinham IDs arXiv pós-abr/2026 (inválidos/alucinados). Descartados. Mantidos só links verificáveis com arXiv IDs reais (2501-2512).

---

## 1. Pretraining efficiency

### S1. Muon is Scalable for LLM Training (Moonshot AI, fev/2025)

- **Link:** [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)
- **Contribuição:** otimizador baseado em ortogonalização via Newton-Schulz escala pra modelos grandes com duas correções (weight decay + ajuste de update scale por parâmetro). ~2x eficiência computacional vs AdamW em regime compute-optimal. Treinaram Moonlight 3B/16B MoE com 5.7T tokens.
- **Por que importa pro Maracatu:** trocar AdamW por Muon no M-80M pode render ~30-50% menos tokens pra atingir mesma loss. Implementação é ~200 linhas de PyTorch. Alto leverage, baixo risco arquitetural.
- **Tier:** S

### S2. Kimi K2: Open Agentic Intelligence (Moonshot, jul/2025)

- **Link:** [arXiv:2507.20534](https://arxiv.org/abs/2507.20534)
- **Contribuição:** MoE 1T/32B ativos. Introduz MuonClip (Muon + QK-Clip pra estabilidade), arquitetura DeepSeek-V3-like com 384 experts. 15.5T tokens sem instabilidades significativas.
- **Por que importa:** QK-Clip é receita barata pra estabilizar Muon em escala. No M-8B pode ser o diferencial entre "treinamento convergiu" e "loss explodiu na iter 30k".
- **Tier:** A

### A3. Insights into DeepSeek-V3: Scaling Challenges and Hardware Reflections (mai/2025)

- **Link:** [arXiv:2505.09343](https://arxiv.org/html/2505.09343v2)
- **Contribuição:** pós-mortem técnico do treino do V3, com análise detalhada de FP8, MLA e choices de hardware. Não é paper "novo resultado", é o documento mais rico sobre o que deu certo e o que quase deu errado em escala.
- **Por que importa:** leitura obrigatória antes de commitar arquitetura final do M-8B/80B. MLA custa 4-7% de perplexidade vs MHA mas reduz KV-cache em ~93%, trade-off crítico pra inference barata.
- **Tier:** A

### A4. Recipes for Pre-training LLMs with MXFP8 (NVIDIA, jun/2025)

- **Link:** [arXiv:2506.08027](https://arxiv.org/html/2506.08027v1)
- **Contribuição:** receita completa pra MXFP8 (block-scaled FP8), incluindo 8B dense treinado em 15T tokens matching BF16 baseline. ~22% redução de tempo, 14% menos peak memory, 19% mais throughput.
- **Por que importa:** MXFP8 é o que Blackwell (B200) vai priorizar. Na escala M-8B+ é 20%+ de economia direta. Kaggle T4 e A100 antigo não suportam bem, então só relevante pós-8B.
- **Tier:** A (esperar escala)

### B5. Scaling Laws for Precision (ICLR 2025)

- **Link:** [PDF Harvard](https://pehlevan.seas.harvard.edu/sites/g/files/omnuum6471/files/2025-03/Kumar_etal_ICLR_2025.pdf)
- **Contribuição:** formaliza trade-offs entre precisão (FP8/FP4), parâmetros e dados. Mostra que precisão mais baixa reduz "effective parameter count".
- **Por que importa:** conceitual, útil pra decidir quando ir pra FP8. Não muda nada no M-80M.
- **Tier:** B

---

## 2. Data curation 2025+

### S6. BeyondWeb: Scaling Synthetic Data for Trillion-scale Pretraining (DatologyAI, ago/2025)

- **Link:** [arXiv:2508.10975](https://arxiv.org/abs/2508.10975)
- **Contribuição:** framework de rephrasing sistemático de documentos web. Um 3B/180B tokens em BeyondWeb supera um 8B/180B em Cosmopedia. Reportam 7.7x training speedup via qualidade de dados.
- **Por que importa pro Maracatu:** provavelmente o paper de maior ROI pra o ladder atual. Investir em rephrasing dos 1.57B tokens que já temos pode render 2-3x melhoria efetiva, mais barato que buscar mais corpus. Embaraçosamente paralelizável via APIs de inferência.
- **Tier:** S

### S7. Ultra-FineWeb: Efficient Data Filtering and Verification (mai/2025)

- **Link:** [arXiv:2505.05427](https://arxiv.org/abs/2505.05427)
- **Contribuição:** pipeline de filtragem via fastText classifier leve, gerando ~1T tokens EN + 120B ZH de qualidade superior a FineWeb-Edu.
- **Por que importa:** pipeline barato (fastText roda em CPU), replicável pra PT-BR. Se combinado com o corpus de 120B PT do item S8, dá pra montar Ultra-FineWebPT sem queimar GPU.
- **Tier:** S

### S8. Building High-Quality Datasets for Portuguese LLMs (set/2025)

- **Link:** [arXiv:2509.08824](https://arxiv.org/html/2509.08824)
- **Contribuição:** 120B token corpus PT-BR a partir de Common Crawl, com pipeline de filtragem industrial. Compete com corpora industriais usados em modelos fechados.
- **Por que importa:** leitura obrigatória. Playbook pro upgrade do corpus Maracatu v3. Com 120B tokens, M-800M deixa de ser sub-Chinchilla e vira viável de verdade. Ladder acelera 6-12 meses se conseguirmos o dataset ou replicarmos o pipeline.
- **Tier:** S

### A9. MixtureVitae: Permissive-First Pretraining Dataset (set/2025)

- **Link:** [arXiv:2509.25531](https://arxiv.org/abs/2509.25531)
- **Contribuição:** corpus 100% permissivo (public domain + licenças permissivas). A 1.7B/300B tokens, supera FineWeb-Edu e aproxima DCLM.
- **Por que importa:** alinha com postura Apache 2.0 do Maracatu. Pra captação via edital público (FINEP, BNDES), ter corpus com trilha de proveniência limpa é ativo jurídico e narrativo enorme.
- **Tier:** A

### A10. Aleph-Alpha-GermanWeb (mai/2025)

- **Link:** [arXiv:2505.00022](https://arxiv.org/html/2505.00022v3)
- **Contribuição:** model-based data curation + synthetic data especificamente pra alemão, língua mid-resource similar em volume ao PT-BR.
- **Por que importa:** caso mais diretamente transferível pra nossa situação (língua europeia mid-resource, lab menor que frontier). Receita replicável com ajustes mínimos.
- **Tier:** A

---

## 3. Post-training barato

### S11. DeepSeek-R1 Distillation Recipe (jan/2025, já na baseline)

- **Link:** [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- **Contribuição:** mostra que SFT puro com 800k trajetórias de raciocínio destiladas de R1 basta pra transformar Qwen-7B num reasoner competitivo. Sem RL necessário no student.
- **Por que importa:** caminho pro M-8B ganhar capacidade de reasoning sem GPU cluster. Dataset open existe (Open-R1, NuminaMath). SFT em M-8B com 800k samples cabe em 1 A100 por alguns dias.
- **Tier:** S (referência canônica)

### S12. REDI: Reinforcement Distillation from Teacher Data (mai/2025)

- **Link:** [arXiv:2505.24850](https://arxiv.org/abs/2505.24850)
- **Contribuição:** sucessor direto de DPO/SimPO pro contexto de distillation. SFT Stage 1 (só positivos) + REDI Stage 2 (positivos + negativos via REINFORCE-style). Qwen-REDI-1.5B atinge 83.1% MATH-500 com apenas 131k trajetórias, igualando R1-Distill-1.5B que usou 800k.
- **Por que importa:** 6x menos dados pra mesma qualidade em distillation de reasoning. Diretamente aplicável ao M-8B. Dá pra testar no M-800M como prova de conceito.
- **Tier:** S

### A13. RLVR Implicitly Incentivizes Correct Reasoning in Base LLMs (jun/2025)

- **Link:** [arXiv:2506.14245](https://arxiv.org/abs/2506.14245)
- **Contribuição:** framework teórico mostrando por que GRPO com rewards binários funciona em base LLMs (sem SFT prévio). Demonstra que o sinal incentiva raciocínio correto cedo no treino.
- **Por que importa:** valida que podemos pular SFT caro e ir direto GRPO no M-8B se tivermos verificadores (math, código). Precisa massa crítica, provavelmente inviável sub-1B.
- **Tier:** A

### A14. Shrinking the Variance: Shrinkage Baselines for RLVR (nov/2025)

- **Link:** [arXiv:2511.03710](https://arxiv.org/abs/2511.03710)
- **Contribuição:** melhora estimação de baseline em GRPO via shrinkage estimators, reduz variância de gradiente.
- **Por que importa:** melhoria de engenharia barata em cima de GRPO. Relevante quando formos fazer RL post-training no M-8B.
- **Tier:** A

---

## 4. Arquiteturas eficientes

### A15. Nemotron-H: Hybrid Mamba-Transformer (NVIDIA, abr/2025)

- **Link:** [arXiv:2504.03624](https://arxiv.org/html/2504.03624v3)
- **Contribuição:** família 8B/56B híbrida Mamba-2 + Transformer (maioria Mamba, algumas camadas de attention). Throughput dramaticamente superior em long context mantendo qualidade.
- **Por que importa:** interessante mas quebra compatibilidade com LlamaForCausalLM, perdemos via HF/Ollama. Não recomendo adotar agora. Vale monitorar se ecosystem suporta bem.
- **Tier:** A (monitorar, não adotar)

### A16. Nemotron 3 Nano: Hybrid Mamba-Transformer MoE (NVIDIA, dez/2025)

- **Link:** [arXiv:2512.20848](https://arxiv.org/html/2512.20848v1)
- **Contribuição:** MoE + Mamba-Transformer hybrid pra agentic reasoning. Direção mais recente da NVIDIA em small-efficient models.
- **Por que importa:** sinal de pra onde frontier vai. Mesma ressalva de compatibilidade HF do anterior.
- **Tier:** B

### B17. Kimi Linear: Expressive Efficient Attention (out/2025)

- **Link:** [arXiv:2510.26692](https://arxiv.org/pdf/2510.26692)
- **Contribuição:** variante de linear attention competitiva com full attention em quality e muito mais barata em long context.
- **Por que importa:** long-context efficiency pro futuro distante. Quebra compat HF.
- **Tier:** B

---

## 5. Long context barato

### S18. LongRoPE2: Near-Lossless LLM Context Window Scaling (ICML 2025)

- **Link:** [ICML 2025 poster](https://icml.cc/virtual/2025/poster/44280)
- **Contribuição:** estende Llama3-8B pra 128k context usando apenas 10B tokens (80x menos que Meta), mantendo 98.5% de short-context performance. Usa evolutionary search pra rescaling de RoPE + mixed-context training.
- **Por que importa:** extensão de contexto barata é exatamente o que um lab solo precisa. 10B tokens de long-context data é viável em RunPod spot por ~R$2-5k. Um "M-8B-long" com 64k-128k context é feature comercial forte (RAG, OAB com contratos longos).
- **Tier:** S

### A19. Rope to Nope and Back Again (jan/2025)

- **Link:** [arXiv:2501.18795](https://arxiv.org/abs/2501.18795)
- **Contribuição:** arquitetura híbrida intercalando camadas NoPE e RoPE. NoPE camadas aprendem retrieval, RoPE camadas aprendem contexto local. SmolLM3 já usa essa ideia (3:1 RoPE:NoPE).
- **Por que importa:** pequena modificação arquitetural (alguns layers sem RoPE) com ganho mensurável em long context. Preserva compat HF se implementado como flag. Candidato a testar no M-800M.
- **Tier:** A

---

## 6. Recipes de modelos pequenos que bateram frontier

### S20. Qwen3 Technical Report (mai/2025)

- **Link:** [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)
- **Contribuição:** família 0.6B até 32B dense + 30B-A3B e 235B-A22B MoE. Qwen3-1.7B base supera Qwen2.5-3B base em >50% dos benchmarks. 36T tokens, 119 línguas. QK-Norm adicionado pra estabilidade.
- **Por que importa:** Qwen3-0.6B-base é o baseline direto pro Maracatu-800M. Se eles com 36T de corpus multilíngue batem modelos 3x maiores, a maior parte do ganho é data scale, não params. QK-Norm é trivial de adicionar ao nosso model.py.
- **Tier:** S

### S21. SmolLM3 (HuggingFace, jul/2025)

- **Link:** [blog oficial](https://huggingface.co/blog/smollm3) + [HF card](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base)
- **Contribuição:** 3B dense, dual-mode reasoning, 6 línguas (incluindo PT), 128k context via YARN, totalmente open (pesos + data mixture + configs). 11.2T tokens staged curriculum. GQA 4 grupos + NoPE a cada 4 layers.
- **Por que importa:** template exato pro Maracatu-800M. Mesma escala de ambição, mesma filosofia open. Já cobre PT nativamente, o que muda o threshold: não basta "ser bom em PT", precisa ser notavelmente melhor que SmolLM3-3B em PT pra justificar existência.
- **Tier:** S

### A22. Gemma 3 Technical Report (mar/2025)

- **Link:** [arXiv:2503.19786](https://arxiv.org/abs/2503.19786)
- **Contribuição:** 1B/4B/12B/27B com knowledge distillation from larger teacher durante pretraining. Gemma3-4B-IT competitivo com Gemma2-27B-IT.
- **Por que importa:** o resultado forte é que distillation-during-pretraining (não só post-training) rende ganhos enormes em scale pequeno. Pro M-800M/M-8B dá pra destilar de um teacher aberto (Qwen3-32B, Llama-3.3-70B) durante pretrain. Custo: 1.5-2x mais caro em compute, mas 30-50% melhor modelo final.
- **Tier:** A

### A23. Phi-4 Technical Report (dez/2024, borderline baseline)

- **Link:** [arXiv:2412.08905](https://arxiv.org/abs/2412.08905)
- **Contribuição:** 14B com training recipe centrado em synthetic data. Reporta 12 épocas em synthetic data sem overfitting.
- **Por que importa:** inspiração pra Phi-style synthetic data em PT-BR. Mas replicar requer teacher forte em PT-BR, que ainda não existe aberto. Viável pós-M-8B.
- **Tier:** A

---

## 7. Línguas subrepresentadas (transferível pro PT-BR)

### A24. Revisiting Multilingual Data Mixtures in LM Pretraining (out/2025)

- **Link:** [arXiv:2510.25947](https://arxiv.org/html/2510.25947v1)
- **Contribuição:** estudo sistemático de mixture ratios entre línguas em pretraining. Refuta várias heurísticas antigas sobre oversample de low-resource.
- **Por que importa:** quando formos adicionar corpora em EN, ES ao PT-BR (crucial pro M-8B ter transfer cross-lingual pra código/science), este é o guia.
- **Tier:** A

### A25. Nemotron-Mini-Hindi 4B (out/2024)

- **Link:** [arXiv:2410.14815](https://arxiv.org/abs/2410.14815)
- **Contribuição:** SLM bilíngue HI+EN, 400B tokens CPT, mix real + synthetic translation data.
- **Por que importa:** caso exato que o Maracatu copia: língua mid-resource + lab nacional + synthetic translation data pra cobrir gaps. Receita direta.
- **Tier:** A

---

## Ressalvas do survey

- **Tucano 2:** busca retornou resultado com ID arXiv inválido (2603.xxxxx). Vale verificar direto em [nkluge-correa.github.io/Tucano](https://nkluge-correa.github.io/Tucano/) se há release nova além do Tucano 1 (nov/2024).
- **muP refinements:** buscas retornaram vários papers com IDs pós-abril/2026 (inválidos). Os concretos de 2025 são pre-prints sobre HyperP (muP + Muon) e CompleteP extensions. Ganho marginal vs muP clássico no nosso regime 20M-8B.
- **Scaling laws totalmente novas pós-Chinchilla:** literatura amadureceu, nada que substitua Chinchilla. Há refinements (inference-aware scaling, precision scaling). Mensagem consensual: treinar muito além de Chinchilla é o default agora (Qwen3-0.6B em 36T tokens = 60000:1 ratio). Valida o que já fazemos no tiny-long.

---

## Top 5 pra ler essa semana

Priorizado por: (impacto no próximo release) x (custo de aplicação) x (leitura densa mas tratável).

1. **SmolLM3 blog + paper card** ([link](https://huggingface.co/blog/smollm3)). Competidor direto no M-800M, suporta PT. Entender decision-by-decision deles antes de escrever `maracatu_800m.yaml`. 1-2h leitura.
2. **Building High-Quality Datasets for Portuguese LLMs** ([arXiv:2509.08824](https://arxiv.org/html/2509.08824)). Muda o teto do que é possível. 2h leitura + avaliar se baixa dataset.
3. **BeyondWeb** ([arXiv:2508.10975](https://arxiv.org/abs/2508.10975)). Maior lever de "fazer mais com menos" que existe hoje. 1.5h leitura.
4. **Qwen3 Technical Report** ([arXiv:2505.09388](https://arxiv.org/abs/2505.09388)). Baseline mental de small dense model bem feito. QK-Norm e três-stage pipeline. 2-3h leitura.
5. **Muon is Scalable** ([arXiv:2502.16982](https://arxiv.org/abs/2502.16982)). Micro-experimento perfeito no M-80M: 5k iters AdamW vs Muon, mesmo seed. Se confirmar 1.5-2x speedup, muda economics de todos os releases. 1.5h leitura + 1 dia experimento.

---

## Recomendação síntese

Resistência à tentação: não aplicar tudo no M-80M. Escala 80M é experimental demais pra ganhos serem conclusivos.

Plano sugerido (ordem de risco crescente):

1. **M-80M treino atual** (semana 0): mantém receita, zero mudanças. Objetivo é publicar release 2 limpo.
2. **Micro-experimento Muon** (semana 1-2, paralelo): rodar 5k iters no tiny com Muon vs AdamW. Se >30% speedup na loss, vira default do M-800M.
3. **M-800M planning** (mês 1-2): incorporar QK-Norm (Qwen3), staged curriculum (SmolLM3), distillation-during-pretrain de um teacher open PT (Qwen3-8B ou Llama-3.1-8B-instruct em PT). Corpus alvo: tentar acesso ao corpus de 120B PT ou replicar pipeline.
4. **Post-training M-8B** (6-12 meses): REDI + distillation de DeepSeek-R1 pra reasoning. Pular SFT tradicional no final.
5. **Long-context M-8B variant:** LongRoPE2 como fine-tuning separado (10B tokens), release como "M-8B-long".

Trade-off explícito: adotar BeyondWeb + Muon + distillation-during-pretrain no M-800M custa ~30-50% mais engineering time (2-3 semanas extra) mas estima-se render equivalente a 2-3x mais tokens/params no modelo final. Se o objetivo é "Maracatu-800M melhor que SmolLM3-3B em PT-BR", não é luxo, é requisito.
