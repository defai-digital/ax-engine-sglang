## 1. Ceiling: 3x MTP lossless is impossible

Arithmetic: speedup over AR = `A / (D+V)`, `A`=accepted/round incl. bonus, `D,V`=draft, verify cost in target-forwards.

Observed (GSM8K): MTP `A=5.02`, 2.59x AR → `D+V=1.94`. DFlash2 `A=5.46`, 3.43x AR → `D+V=1.59`. DSpark beats MTP with lower `A` only because its engine round is cheaper.

Block 8, realistic best at T=1.0: `A=5.5-6.0`, `D+V=1.6-2.0` → 2.8-3.8x AR → **1.1-1.45x MTP**. Absolute fantasy bound: perfect `A=9`, cost 1.5 → 6.0x AR → 2.3x MTP.
Block 16: `A=6.5-7.5`, cost 2.5-3.5 → 1.9-3.0x AR → worse per second. Bigger blocks lose; verify cost grows ~0.15/forward per token.
3x MTP ≈ 7-8x AR needs `A≈12-15` at cost ≤2, i.e. 12+ perfect tokens every round at temp 1.0. No lossless design does that. Tree drafting (+0.5-1.0 `A` for +0.2-0.4 cost) is the only lever left, worth ~10-20%.

Verdict: target **1.3-1.4x current MTP at conc 1**, parity at conc 32. Anything promising 2x+ lossless on this target is mismeasured.

## 2. Track A: keep 1+4, cut or shrink 2+3

1. **Fuse target states: yes, mandatory.** This is the whole gap (your 2.15 free-running never saw the target). Tap 5 layers like DSpark.
2. **2-3 refinement passes: no as proposed.** Each pass multiplies `D` (~0.2-0.4x each) for +0.2-0.5 `A`. One strong parallel pass + tree beats two thin passes. Keep at most one cheap rescore of positions 4-7.
3. **Adaptive length: low value.** Verify cost is sublinear; fixed k=7-8 with early-exit on low confidence captures 90% of the gain. Do not build a calibrated head first.
4. **Data+loss: yes, but fix it.** Prefix-weighted CE + top-k KL is right; add on-policy training (train on your own drafts, not teacher forcing) and an acceptance-weighted loss. Pure teacher forcing is why plan-mode showed 2.0→1.07 free-running.

Missing, in order: **tree/multi-candidate verification** (biggest win), reusing the frozen MTP head as position-0 predictor, own small KV for draft tokens only, no DeltaNet-state prediction (tapped states already encode it).

Recommended arch: tap layers 5/19/33/47/61 → project 5120→2048; 6-layer GQA decoder, d=2048, FFN 8192, 16Q/4KV heads, RoPE; inputs = tapped-h mix + accepted-token embeds; heads: token logits + 2-way tree expansion at pos 0-3, greedy rest; ~1.2-1.6B params; `D≈0.25x` per pass.

## 3. Real NCM: no place

VQ codes compound error (your evidence: 1.07/round self-predicted), add codebook + detoken latency, and buy nothing under a lossless token verifier. The only "coarse" that helps is first-pass token argmaxes as conditioning for a rescore — that is multi-pass speculative decoding, not NCM. Keep the name for continuity if you must, but do not put codes on the critical path. At most: 256-entry auxiliary planning loss, never decoded free-running.

## 4. Build vs reuse: reuse

Do not train from scratch. Fastest path beating DFlash2: SGLang + Apache-2.0 DFlash2 weights + SpecForge/z-lab/dflash code → reproduce 1.32-1.48x table (days) → fine-tune on thinking+chat mix with tree + acceptance loss. Your salvageable assets are the simulator, verify timer, and harness — not the drafter. Genuinely new: tree-DFlash or MTP-fused drafter with +10-15% over DFlash2 on code+chat, same-engine gated.

## 5. Data/compute on 2x B200

Need **30-80M target tokens** (EAGLE/DSpark regime). 27B bf16 ≈54 GB fits one B200; batched offline gen ≈800-1500 tok/s/GPU → 50M tokens in ~6-17 h. Never cache hidden states: 50 KB/tok × 50M = 2.5 PB. Run frozen target online (forward ~10 ms, batched train step 0.5-1.5 s/4K toks). 10-20K steps × 8K toks ≈ 30-80 GPU-h on 2x B200. Budget fits with wide margin; spend rest on eval/ablations.

## 6. Track B: do not start

A block-emitting 27B sacrifices losslessness — your only edge over open diffusion models — and needs 5-50B tokens to recover HumanEval-strict quality. Smallest test if insisted: LoRA-r32 on last 8 layers + 8-token block head, 200M toks. Kill if >1.5-pt HumanEval/MBPP/GSM8K drop or <2x MTP end-to-end. Gate on Track A landing first.

## 7. Engine: SGLang

SGLang for benchmark and integration: DSpark/DFlash native, faster speculative kernels, tree verify ready. vLLM DFlash is PR-only and its MTP path is slower. Keep vLLM only for Phase-0 continuity numbers.

## 8. Plan with gates

1. **Reproduce SOTA** (SGLang, 1x B200, bf16): AR vs tuned MTP k=1..7 vs DSpark v2 vs DFlash2, conc 1+32, code+chat. Gate: match published ±10% or stop and debug harness.
2. **Fine-tune DFlash2 + tree verify** on 30-50M thinking/non-thinking toks. Gate: ≥1.25x tuned MTP conc 1, ≥AR at conc 32, else kill Track A variants.
3. **MTP-fused + on-policy v2** (recommended arch). Gate: ≥1.35x MTP conc 1 on code+chat, no conc-32 regression; else ship step-2 model.

First three experiments: (a) k-sweep 4/7/8/12 measuring `A`, `D+V`, tok/s; (b) tree width 1 vs 2x4-prefix at fixed k=7; (c) teacher-forced vs on-policy fine-tune, 2K steps each, free-running `A`.
