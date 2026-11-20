**Reopen NCM as a bounded research track. The existing experiment does not reject it. Its strongest opportunity is improving long-block agreement through shared structure and correction; sampled decoding is promising, but latent coherence alone does not guarantee acceptance.**

1. **What can win**

Shared coarse variables can coordinate distant choices—identifiers, syntax, reasoning structure—before fine levels resolve wording. Residual refinement can repair those choices; short early levels reduce arithmetic and enable conditional computation.

What does **not** establish an advantage: semantic similarity, tokenizer reconstruction alone, Sudoku accuracy, or speedups over diffusion. Exact-prefix acceptance rewards the teacher’s particular continuation. Multiple refinement passes also reread weights, so short levels are not proportionally cheap.

DFlash2 already conditions its selector on the previously selected token; its limitation is restricted dependence, not complete independence. [Implementation](https://docs.nvidia.com/nemo/automodel/v0.5/nemo-automodel/nemo_automodel/components/speculative/dflash/draft_qwen3_dflash2)

2. **Sampling: your premise is right, with an important qualification**

For tractable conditional proposals, \(q=p\) gives acceptance one under rejection sampling, including the required residual correction after rejection. [Algorithm](https://arxiv.org/html/2302.01318)

For NCM:

- **Independent exact match:** sample target tokens, retain matching draft prefixes, emit the first mismatch. Lossless without \(q\), but even \(q=p\) gives collision probability \(\sum_x p(x)^2\), not one.
- **Conditional rejection:** sample latent \(z\), then use an explicit stochastic decoder \(q_z\). Holding \(z\) fixed, apply rejection against its actual token conditionals. Marginalizing over \(z\) is unnecessary. However, nearly deterministic \(q_z\) can have poor overlap with \(p\), despite an excellent marginal mixture.
- **Gumbel coupling:** equality of *token conditional distributions* yields identical samples with shared noise. Equality of a latent model’s joint distribution does not imply the same sampling map. A density-free implementation can compare against target draws using fixed, position-keyed noise retained across retries; arbitrary reuse/redrawing after rejection needs a correctness proof. [Coupling results](https://openreview.net/pdf?id=GJbrk55Bt8)

Thus, **“sample concepts, make wording deterministic” can actually hurt sampled acceptance**.

3. **The negative result was not fair**

The [report](/Volumes/Ext16TR0/code/ax-engine-sglang/results/phase2_ablation/REPORT.md) establishes failure of that configuration, with learning curves still improving. Gold-code teacher forcing is legitimate training; evaluating with gold codes exaggerates deployment performance.

Also, current [drafter.py](/Volumes/Ext16TR0/code/ax-engine-sglang/phase2/drafter.py:270) contains corruption/rollouts and recomputes corrective labels. That differs from the paper’s **clean, unchanged labels**; historical run settings need checking.

Minimum credible test: held-out tokenizer accuracy ≥99.9% overall and ≥99.95% code, whole-block reconstruction measured; target-hidden conditioning; 6/8/12-level comparisons; decreasing corruption averaging 0.5; 30–100M target-generated tokens spanning all workloads; approximately 20k–50k updates **and validation plateau**, not a step-count ritual. Compare matched-data DFlash2 and continuous refinement. Continue for ≥10% measured speed improvement; stop this implementation after convergence if every latency/acceptance operating point is dominated.

4. **My NCM-Draft**

Start at **32 tokens**, deploy 16 adaptively; defer 64. Use a shared 100–200M transformer core, discrete residual VQ levels **[1,2,4,8,16,32]**, plus a small token decoder. Keep a continuous-residual version as the essential control.

Project cached target states from DFlash2’s existing taps into prefix memory available at every level; never use future target states. Train token-distribution distillation alongside code prediction.

Greedy verification uses target argmax. Sampling initially uses explicit \(q_z\) rejection; independent exact match is the correctness reference.

Train intermediate exits. When expected additional accepted tokens no longer repay another level, decode a shorter prefix or fall back. Low confidence alone is insufficient: some uncertain blocks benefit most from refinement.

Planning estimates, **not measurements**: 4–6 generator passes plus decoding; drafting 0.4–0.9 single-target-step times, verification 1.3–2.0, total 1.7–2.9. Forecast emitted tokens/round: greedy code 12–18, math 9–14, chat 4–7; sampled thinking 5–9. These ranges include failure to beat DFlash2.

5. **Long blocks and faithful implementation**

Yes, 16–32 is the best initial opportunity. But logarithmic **depth** is not logarithmic cost: geometric levels contain \(O(L)\) positions, dense attention can cost \(O(L^2)\), and verification grows too.

Illustrative greedy-code hypotheses—not fitted estimates—are:

| Prefix survival \(P(A\ge j)\) | j=8 | j=16 | j=32 |
|---|---:|---:|---:|
| Extended one-pass DFlash2 | .85 | .30 | .02 |
| Successful NCM-Draft | .85 | .55 | .12 |

Measure survival curves directly; marginal token accuracy hides correlated failures.

Copy clean-label corruption, nested dropout, cumulative residual conditioning, and block-diagonal level attention. Start loss weighting at \(\alpha=1\); ablate 0. Neither level count nor corruption optimum transfers automatically. [Paper](https://arxiv.org/html/2610.14162v1)

For 248k vocabulary, share/project the token head and avoid full-vocabulary projection at every level. Preserve rare-token fidelity. DeltaNet verification must correctly checkpoint/restore recurrent **and convolutional** states at the accepted boundary. Count actual passes: released-code timing discussion identifies an NFE counting discrepancy. [Author discussion](https://github.com/HazyResearch/ncm/issues/9)

6. **Hybrid ranking**

1. NCM extension for positions 9–32: strongest immediate economic case.
2. Gated multiscale refinement after DFlash2: easiest controlled comparison.
3. NCM selector replacement: useful hypothesis, higher sampling/integration risk.
4. Lossy thinking-only generation: separate research objective; changes downstream answers and violates Track A.

7. **Subjective probabilities after serious tuning**

Against equally tuned baselines on identical workloads:

| Workload | ≥10% over DFlash2 | ≥1.5× MTP | ≥3× MTP |
|---|---:|---:|---:|
| Greedy code/math | 40% | 55% | 5% |
| Sampled thinking | 30% | 40% | 3% |
| Chat | 20% | 30% | 1% |

These are judgments, not statistical estimates.

8. **First three experiments; budgets are aggregate GPU-hours**

- **6 GPU-hours:** audit sampling correctness, emitted/accepted/bonus accounting, tuned MTP, and verification widths 8/16/32 at concurrency 1/32. Proceed only with greedy parity and a demonstrated sampling-correctness argument.
- **18 GPU-hours:** tokenizer plus oracle-code ceiling, comparing discrete and continuous representations. Gate: reconstruction thresholds above and oracle throughput ceiling ≥1.25× DFlash2.
- **18 GPU-hours:** matched-data pilots of faithful NCM, continuous refinement, and DFlash2 extension. Proceed if measured speed improves ≥10%, with confidence intervals excluding parity, and removing the hierarchy erases the gain.

The pilots screen investment; unconverged failure is not a rejection of NCM.
