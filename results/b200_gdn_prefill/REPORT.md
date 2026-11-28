# B200 · Qwen3.8-27B NVFP4: FlashInfer GDN prefill and one-at-a-time DFlash admission (2026-11-28)

Qwen3.8-27B NVFP4 (ModelOpt mixed precision) with the DFlash2 drafter on one B200 (SM100, 183,359 MiB,
driver 595.112.07). SGLang 0.5.23, torch 2.14.1+cu130, FlashInfer 0.7.2, FlashInfer cute-dsl FP4 GEMMs,
`trtllm_mha` attention, FlashInfer autotune on, 8,192-token prefill chunks, prefill CUDA graphs on.
Stock vLLM 0.34.0rc1.dev1105+g2f170a0fe (torch 2.14.0, FlashInfer 0.7.4) on the same pod. Workload,
vLLM settings and pooling as in `results/h200_humming` (first sweep of every server dropped; n = servers).
RTX PRO 6000 rows: one RTX PRO 6000 Blackwell Workstation (SM120), same software, same-pod vLLM.

## Cause

SGLang 0.5.23 runs Gated-DeltaNet prefill with FlashInfer only on SM90 with an fp32 state or on SM100
with a bf16 state (`flashinfer_gdn_prefill_default`). The DFlash routes keep an fp32 state, so the B200
ran the FLA Triton chunk kernels: about six eager launches per layer with 20-80 µs gaps between them.
In an eight-client math profile the 20-100 µs idle gaps took 80 ms of a 444 ms span against 26 ms for
vLLM, which uses FlashInfer GDN prefill. `--linear-attn-prefill-backend flashinfer` reads the fp32 pool
as given; verify stays on Triton (SGLang moves verify to FlashInfer only with the FlashInfer decode).
The dispatcher logged `decode=TritonGDNKernel, extend=FlashInferGDNKernel, verify=TritonGDNKernel`.

## Results

Output tok/s, code / math / chat. "admit 1" = `--min-free-slots-delay 1`, "FI prefill" =
`--linear-attn-prefill-backend flashinfer`. First token: p50 / p90.

| B200 route, eight clients | n | tok/s | Against vLLM | First token |
|---|---:|---:|---:|---:|
| AX GDN replay 8 | 2 | 2,418.5 / 2,266.3 / 1,464.8 | −9.1 / −20.5 / −19.5% | 0.079 / 0.186 s |
| AX GDN replay 8, admit 1 | 2 | 2,375.9 / 2,398.8 / 1,496.3 | −10.7 / −15.9 / −17.8% | 0.083 / 0.105 s |
| AX GDN replay 8, admit 1, FI prefill | 2 | 2,583.9 / 2,660.5 / 1,645.8 | −2.9 / −6.7 / −9.6% | 0.065 / 0.078 s |
| AX tree 32 | 2 | 2,427.7 / 2,074.3 / 1,342.1 | −8.7 / −27.3 / −26.2% | 0.086 / 0.154 s |
| AX tree 32, admit 1 | 1 | 2,115.7 / 2,104.6 / 1,381.8 | −20.5 / −26.2 / −24.1% | 0.082 / 0.098 s |
| AX tree 32, admit 1, FI prefill | 2 | 2,414.5 / 2,503.5 / 1,502.4 | −9.2 / −12.2 / −17.4% | 0.072 / 0.090 s |
| Stock vLLM DFlash2 | 2 | **2,660.1 / 2,851.6 / 1,819.6** | | 0.084 / 0.111 s |

| B200 route, one client | n | tok/s | Against vLLM |
|---|---:|---:|---:|
| AX GDN replay 8, admit 1 | 2 | 711.5 / 600.5 / 386.2 | −15.1 / −13.7 / −14.5% |
| AX GDN replay 8, admit 1, FI prefill | 2 | 711.7 / 643.8 / 404.8 | −15.1 / −7.5 / −10.4% |
| AX tree 32, admit 1 | 1 | 851.0 / 727.5 / 448.4 | +1.5 / +4.6 / −0.7% |
| AX tree 32, admit 1, FI prefill | 2 | **932.5 / 753.3 / 458.0** | **+11.2 / +8.3 / +1.4%** |
| Stock vLLM DFlash2 | 2 | 838.4 / 695.6 / 451.6 | |

| RTX PRO 6000 route | Clients | n | Admit 1 | Admit 1, FI prefill (n = 1) |
|---|---:|---:|---:|---:|
| AX GDN replay 8 | 1 | 2 | 329.4 / 267.1 / 174.9 | 313.0 / 273.3 / 171.7 |
| AX GDN replay 8 | 8 | 2 | 1,194.8 / 1,313.8 / 790.1 | 1,384.1 / 1,375.7 / 751.4 |
| AX tree 32 | 1 | 2 | 431.4 / 363.4 / 209.5 | 440.8 / 350.5 / 199.6 |
| AX tree 32 | 8 | 2 | 1,151.8 / 1,216.0 / 684.9 | 1,251.3 / 1,214.6 / 689.9 |

- **FlashInfer prefill on B200** raised replay 8 at eight clients by 9-11% and the tree at one client by
  2-10% (code 932.5 tok/s, 11% ahead of vLLM), and cut the eight-client first-token p90 to 0.08-0.09 s.
- **Admission** cut the eight-client first-token p90 from 0.15-0.19 s to 0.10 s; throughput moved −13 to
  +6%, within the eight-client pass-to-pass spread of these lanes (up to 13% on code).
- **RTX PRO 6000 (SM120)** showed no consistent gain from FlashInfer prefill: one-client chat −2 to −5%, eight
  clients −5 to +16%. It keeps Triton. Admission there: eight-client math and chat +3 to +7%, code −1 to
  −13% (code spread 1,079-1,338 between the two replay servers), first-token p90 0.16-0.27 → 0.11-0.12 s.
- **Remaining B200 gap** is per decode round (equal tokens per verify; chat decode 0.50 s per request
  against vLLM's 0.45 s). Three independent source reviews (Codex, agy, Grok) placed it in the replay
  fold after acceptance (about 0.6 ms per round, vLLM selects a saved state instead), the verify conv
  writing its intermediate windows (about 0.3 ms), and the RMSNorm + FP4/FP8 quantization vLLM fuses
  (about 0.3-0.4 ms). These need new kernels; no switch removes them.

## Launch default

The launcher now sets `--linear-attn-prefill-backend flashinfer` for DFlash and tree launches of a
Gated-DeltaNet target on SM100 with CUDA 13 or newer, when the user set neither that flag nor
`--linear-attn-backend`, the prefill chunk is 1-8,192 tokens (SGLang's own SM100 domain), and
deterministic inference, dynamic chunking and the parity profiles are off.
