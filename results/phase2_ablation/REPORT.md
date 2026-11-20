# Phase 2 architecture ablation (2026-11-14)

Small-scale screen, 1x RTX PRO 6000. Each drafter: frozen Qwen3.5-0.8B encoder + 6-layer decoder,
2,000 steps, 128 blocks per step. Data: 4,500 prompts answered greedily by **Qwen3.5-0.8B itself**
(a stand-in verifier), so the first token is easier here than it will be with Qwen3.6-35B.
Metric: free-running accepted prefix length on held-out responses (tokens accepted per draft, before
the verifier's own token).

| Variant | Passes | Accepted prefix @500 / 1000 / 1500 / 2000 steps |
|---|---|---|
| direct: 32 tokens in one parallel pass | 1 | 0.07 / 0.13 / 0.18 / 0.20 |
| chunks [1,1,2,4,8], uniform loss | 5 | 0.27 / 0.31 / 0.41 / 0.45 |
| chunks [1,1,2,4,8], prefix-weighted loss (0.8^j) | 5 | 0.31 / 0.42 / 0.50 / 0.65 |
| **chunks + prefix loss + encoder-state init** | 5 | **1.35 / 1.73 / 1.94 / 2.15** |

Earlier run (before the rework), code plan + teacher forcing, 1,500 steps: 2.0 accepted tokens
teacher-forced but 0.07 free-running (the gold codes leak the target).

Not run: plan + rollout training, plan + teacher forcing under the new metric, chunks [1,1,2,4].
They were queued on the RTX PRO 6000 pod, which was released before they started.

Reading: token dependence across chunks, prefix weighting and starting from the encoder's own
next-token state each help, and the last one helps most. All curves were still rising at 2,000 steps.
