# How AX Engine speeds up SGLang

In autoregressive decoding (AR), each request needs a target-model forward
pass for every new token. Speculative decoding proposes several tokens ahead
and lets the target check them in one pass. When the proposals agree with
the target, the server can emit several tokens before it needs another
target pass. AX keeps the target checkpoint and extends the verification
paths inside SGLang, which continues to handle scheduling, model loading and
CUDA execution. [Runtime boundaries](README.md#runtime-boundaries)

## Spend verification work where it pays off

For an eligible single request on dense Qwen, AX verifies a tree of candidate
continuations in one pass. A tree can cover several plausible next tokens, so
the target can keep accepting tokens along another branch when the drafter's
first choice diverges. On Qwen3.8-27B, DFlash2 supplies scores for transitions
between candidates, which AX uses to decide which branches fit the node budget.
Each candidate attends to the committed prefix and its own branch history.
Only the accepted path advances the request's cache and recurrent state.

When multiple requests are active, AX verifies a linear chain per request.
Batching already lets one target pass serve several requests, and a chain
avoids spending verification rows on extra branches for each request. The
chain can also use an attention backend tuned for batched decoding. On RTX
PRO 6000, the dense 27B tree delivered 1.26× stock DFlash2 throughput with
one active request. At concurrency eight, the two routes were within about
1%. [Round and backend design](decisions/2026-11-16-dual-attention-backend.md) ·
[RTX benchmark](performance.md#rtx-pro-6000--qwen38-27b-nvfp4)

## Fit the verifier to the model

Flash Next already has multi-token prediction (MTP). Its sparse attention
indexer compresses keys in groups of four. In stock SGLang 0.5.23, later
positions in a speculative window could overwrite keys before compression
read them. AX enlarges the per-request ring buffer that holds those keys,
which preserves them for longer verification windows. The wider MTP7
configuration improved code and math throughput in the B200 run, while stock
MTP3 led on chat. A longer window helps when the extra accepted tokens earn
back the cost of checking them.
[Ring fix](decisions/2026-11-17-qwen38-flash-next-feasibility.md#8-the-qsa-ring-fix-ax_qsa_ring-ax-engine-sglang-ax_qsapy-2026-11-17) ·
[Flash Next benchmarks](performance.md#qwen38-flash-next-nvfp4--mtp)

## Check throughput and output behavior

The target decides which tokens to emit on the configured numerical path.
Batched verification and single-token AR can use different kernels and
produce slightly different logits. On RTX PRO 6000 with dense NVFP4, the
opt-in parity profile reproduced the AR token sequences on all ten greedy
prompts when both routes used the same numerical settings. That profile cost
throughput. We judge speed using matched end-to-end comparisons with AR and
stock speculation where available, and report greedy token identity and
sampled distribution checks for the tested configurations.
[RTX parity report](reports/2026-11-19-rtx-nvfp4-parity.md) · [Serving performance](performance.md)
