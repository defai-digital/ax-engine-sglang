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
chain can also use an attention backend tuned for batched decoding.

On an RTX PRO 6000 with SGLang's prefill CUDA graphs, the 32-node tree led
stock DFlash2 and stock vLLM DFlash2 on every single-request workload:
464.8 / 389.5 / 222.0 output tok/s against 337.5 / 273.6 / 177.9 and
317.2 / 292.0 / 182.9 on code / math / chat. At eight clients,
accepted-prefix GDN replay led, at 1,464.4 / 1,389.7 / 835.2. The tree and
chain route stayed ahead of stock on code and math and was slower on chat.
[Round and backend design](decisions/2026-11-16-dual-attention-backend.md) ·
[Prefill-graph rerun](../results/rtx_pro_6000_prefill_graph/REPORT.md)

## Fit the verifier to the model

Flash Next already has multi-token prediction (MTP). Its sparse attention
indexer compresses keys in groups of four. In stock SGLang 0.5.23, later
positions in a speculative window could overwrite keys before compression
read them. AX enlarges the per-request ring buffer that holds those keys,
which preserves them for longer verification windows.

MTP7 with that wider window led code, and usually math, on B200, RTX PRO
6000, two B300 GPUs, and Jetson Thor. Stock MTP3 led chat on most of those
runs. On one Thor, an adaptive policy that picks the draft length from
observed acceptance raised code, math, and chat throughput together.
A longer window helps when the extra accepted tokens earn back the cost of
checking them.
[Ring fix](decisions/2026-11-17-qwen38-flash-next-feasibility.md#8-the-qsa-ring-fix-ax_qsa_ring-ax-engine-sglang-ax_qsapy-2026-11-17) ·
[Flash Next benchmarks](performance.md#qwen38-flash-next-nvfp4--mtp)

## Check throughput and output behavior

The target decides which tokens to emit on the configured numerical path.
Batched verification and single-token AR can use different kernels and
produce slightly different logits. The default speed profile reports
end-to-end throughput against AR and stock speculation. The opt-in verified
profile forces the parity numerics and starts only on a route with a
recorded token-for-token match.

On RTX PRO 6000, that match covers AR and the width-eight DFlash2 chain:
every sequence equalled the serialized AR reference at 128 tokens (1,000 of
1,000) and at 1,024 tokens (120 of 120). The 16-node tree matched at 128
tokens and diverged on two of ten prompts at 1,024 tokens, so the profile
refuses the tree. `ax-engine-sglang doctor` applies these rules before any
weights load.
[Verified profile](reports/2026-11-21-verified-profile-gate.md) · [Serving performance](performance.md)
