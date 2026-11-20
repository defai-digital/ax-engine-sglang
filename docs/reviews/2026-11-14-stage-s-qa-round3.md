# Stage S code QA, round three — 2026-11-14

Reviewers (read-only): gpt-6.1-sol (codex), Qwen 4 Max preview and GLM 5.3 (ax-code), muse. All four
confirmed the round one and two fixes; the findings below are new. Dispositions first, reviews below.

| Source | Finding | Disposition |
|---|---|---|
| muse 1 | `num_epochs: 2` capped training below `STEPS` | Fixed: `num_epochs: 1000000`, `max_steps` is the only limit |
| codex 1, Qwen 1 | Manifest lacked the target identity (resume after changing `TARGET` mixed answers) | Fixed: `--target` is required and part of the manifest |
| codex 2 | Conversion/capture provenance only checked when the capture stage ran | Fixed: checked whenever s1, smoke or train consume them |
| codex 3 | A server already on `PORT` could answer the health check | Fixed: the port must be free before launch |
| codex 4 | `--verify` trusted the marker | Fixed: every inserted snippet must be present (tested on a fake checkout) |
| codex 5, Qwen 4 | Mixed sweeps gave only DSpine the radix cache | Fixed: `run_baselines.sh` refuses a mixed sweep without `RADIX_CACHE=1`; README corrected |
| codex 6, GLM 1, muse 2 | Long prompts could be truncated or dropped by capture | Fixed: prompts over 1400 tokens or containing template markers are not used |
| codex 7 | Whitespace-normalised prompt keys merged code prompts that differ in indentation | Fixed: exact text |
| codex 8 | Local target directories failed in the evaluator | Fixed |
| GLM 2 | Train/eval fingerprints missed the rest of SpecForge and driver literals | Fixed: `code:$SPECFORGE/specforge` (content of all .py files) and `MAX_LENGTH` are fingerprinted |
| GLM 3, Qwen 3 | Non-atomic manifest write | Fixed |
| Qwen 2 | Torn last JSONL row broke resume | Fixed: incomplete rows are cut off (tested) |

First end-to-end run on the pod (before these fixes, 200 train rows per domain): released
Qwen3-8B DSpine-b16, offline accepted length 5.67 on 900 held-out prompts (code 6.40, math 12.12,
chat 3.80; unary 5.09), greedy consistency 0.9906, no skipped rows; val 5.65. The training smoke
stopped at checkpoint save because the RAM disk was full.


---

## codex

Eight concrete bugs remain. Round-two resume and patch-verification fixes, and round-one cache invalidation, are incomplete in the scenarios below.

1. **[P1] Interrupted generation can still mix targets — [gen_data.py:126](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/gen_data.py:126).** The manifest records `model`, but the driver always passes `--model qwen`, regardless of `TARGET`. Interrupt generation before `data/FINGERPRINT` exists, change `TARGET` to another compatible checkpoint, and resume: the manifest matches, existing answers are retained, and new answers come from the new target. The driver then publishes the mixed dataset under the new target’s fingerprint. **Fix:** persist and validate `DATA_FP`, including the actual target identity, before the first request. Round-two finding #2 remains partially unfixed.

2. **[P1] Omitting capture bypasses upstream provenance checks — [run_stage_s.sh:104](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/run_stage_s.sh:104).** Conversion and capture fingerprints are checked only inside `if has capture`. After completing a run, changing `DRAFT` and invoking `STAGES=train` leaves `train_init`, variants, and run fingerprints unchanged; previously trained results can be returned successfully despite the requested drafter changing. Likewise, training and evaluation consume feature directories without checking their capture markers against the current target/data. **Fix:** validate conversion and capture provenance whenever their outputs are consumed; let stage selection control execution, rather than validation. Round-one finding #8 remains incomplete.

3. **[P1] Readiness can select an unrelated server — [run_stage_s.sh:82](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/run_stage_s.sh:82).** If another server already occupies `PORT` and serves the alias `qwen`, its `/health` immediately returns 200 while the newly launched process is still importing or loading weights. Generation then uses the existing server—even after the new launcher fails to bind—and can publish answers from the wrong model under the intended target’s fingerprint. **Fix:** reject an occupied port before launch, as `run_baselines.sh` already does, and establish that readiness belongs to the launched process.

4. **[P2] `--verify` still accepts partially patched files — [patch_specforge.py:113](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/patch_specforge.py:113).** Any current-version marker skips verification of every replacement in that file. For example, reverting the auxiliary-loss insertion while retaining the telemetry insertion leaves a valid marker in `algorithms/dspine/model.py`; verification succeeds, but `next_pred` trains without its auxiliary objective. I reproduced the verifier accepting files containing current markers but missing edits using in-memory file substitutes. **Fix:** verify every complete replacement snippet, or expected patched-file hashes. The version marker fixes old-version detection, but round-two finding #6’s partial-patch case remains incomplete.

5. **[P2] Default mixed-mode benchmarks use unequal radix-cache policies — [serve_sglang.sh:32](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/serve_sglang.sh:32).** With `RADIX_CACHE` unset, `MODES="ar dspine"` enables caching only for DSpine. The benchmark repeats the same prompts across concurrency levels without flushing the cache, so later DSpine measurements benefit from cached prefixes while AR recomputes them. This makes throughput comparisons unequal even on the same SGLang version. **Fix:** make the benchmark driver choose one cache policy for the entire comparison; suites containing DSpine must enable caching for every arm, or reject incompatible settings.

6. **[P2] Legitimate truncation filtering makes capture fail — [run_stage_s.sh:122](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/run_stage_s.sh:122).** Prompt loading permits up to 6,000 characters, which can exceed 2,048 tokens for code or numeric text. When truncation removes the assistant span, DSpine’s eligibility filter drops that row. Capture can therefore finish correctly with fewer files than JSONL rows, but the driver rejects it; retrying then hits the nonempty unfinished-directory guard. The evaluator’s equal-count check has the same assumption. **Fix:** preserve a manifest of retained source IDs and validate against it, or filter prompts using the exact capture tokenization before generation. The supplied provider registers this filter; [upstream preprocessing applies it after truncation](https://github.com/sgl-project/SpecForge/blob/main/specforge/data/preprocessing.py).

7. **[P2] Decoded prompt matching rejects valid inputs — [eval_accept.py:156](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/eval_accept.py:156).** `prompt_key()` collapses all whitespace, so distinct Python prompts differing in indentation become identical keys and trigger “two rows share the prompt.” This can occur within one split: the two indentation variants of `Explain Python example 5:\nif x:\n    y()\nz()` both hash into evaluation buckets. Separately, line 240 truncates a user prompt containing literal `<|im_end|>`, making lookup fail. Both cases were reproduced. **Fix:** carry source IDs through capture into records or a manifest; avoid recovering identity through whitespace normalization and delimiter splitting. Round-two finding #3’s replacement remains incomplete.

8. **[P2] Local target checkpoints cannot be evaluated — [eval_accept.py:72](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/eval_accept.py:72).** Generation, capture, tokenizer loading, and training accept a local `TARGET`, but `load_target_tables()` unconditionally calls `snapshot_download(target)`. An absolute checkpoint directory raises `HFValidationError`; I reproduced this with the installed Hub client. Thus a local-target run fails at S1 or post-training evaluation. **Fix:** use the directory directly when it exists, and call `snapshot_download()` only for Hub repository IDs.

Checked and found correct, apart from those exceptions:

- **Q1:** Fingerprint assignments propagate failures; digest validation, unknown-arm rejection, JSON argument grouping, smoke-override isolation, `! has data`, and `FEAT_EVAL`/`FEAT_VAL` routing are correct. Signal/exit traps cover normal cancellation and server-group cleanup.
- **Q2 and Q7 walk:** Greedy-segment boundaries, `base` indexing, verifier bonus, clipping, and the test reference are correct for the per-segment metric. Both decoders and all arms use identical target-derived boundaries. Think-scaffold skipping, domain pooling, and limit/skip accounting are correct for successfully matched rows.
- **Q3:** Injection before `pre_injection` preserves refinement conditioning and zero-gate parity. Source windows remain causal; the verified zero-padding convention makes length calculation valid. The stale donor is deterministic and causal. Auxiliary targets are detached, the loss is added once, and adapter/next-prediction parameters and inputs match.
- **Q4:** Hash buckets prevent split overlap; text IDs and manifests detect changed selections and recorded settings. Minimum-completion/error handling and the default system prompt are correct. Actual target identity is the exception in finding 1.
- **Q5:** Variants retain identical base tensors; zero gates, initialization seeds, and configured training seed/data-order inputs are matched. Equal-size rewrites now affect directory fingerprints through mtime.
- **Q6:** The bootstrap correctly resamples paired prompt clusters and computes the ratio of pooled accepted lengths.
- **Q7–Q8:** Old patch versions and wholly unpatched files are rejected. DSpine correctly keeps radix caching enabled. The two pod-verified facts were accepted and not re-raised.

`test_concept.py` and `test_eval_walk.py` passed; the walk also passed 10,000 randomized reference cases. No files were changed and no GPU runs were performed. Full patched-SpecForge execution was unavailable; shared capture filtering was cross-checked against upstream because that helper is absent from the supplied bundle.

---

## qwen

# Stage S 第三輪 QA（唯讀）— 2026-11-14

**審查限制（先說明）：** 本 session 的 bash/monitor 在唯讀模式下被拒，無法跑 shell、git log 或重跑 CPU 測試（此 Mac 無 torch，第二輪已確認）；`/private/tmp/.../sf/` 未 patch SpecForge 參考副本的讀取也被拒（兩次）。因此所有結論來自信件靜態 close-reading + 手動 trace；walk 參考實作與 `greedy_segments` 我以手推逐 case 驗證，但 `test_eval_walk.py`／`test_concept.py`／`test_patch.py` 未在本機重跑。SpecForge 內部行為（`o.hidden_states` 形狀、`denominator` 正規化、position_ids layout、截斷方向）只能依 patch anchor + 測試 + 第一輪 pod 結果推斷。已用 `register_finding` 記錄 4 個發現（workflow=qa）。

---

## 發現的 bug（依嚴重度排序，共 4 個）

**1. [MEDIUM，第二輪 codex#2 仍未修完] resume manifest 不含 target 身分 → 中斷後換 TARGET 會無聲混入兩個 target 的答案** — `track_a/spine/gen_data.py:126-130`（finding `c1fb6b0fcb432383`）。
第二輪宣稱「換 target 不會复用舊答案」，但 manifest 的 `settings.model` 存的是 `args.model`，而 driver 在 `run_stage_s.sh:87` 傳的是 served alias `--model qwen`，**不是 `$TARGET`**。含 `$TARGET` 的 `DATA_FP` 只在「data FINGERPRINT 已存在」時才被 `cached()` 比對；中斷 resume 時 marker 不存在，`cached()` 在 `[[ -f $1 ]] || return 1` 就直接回 1、根本没比 `DATA_FP`（`run_stage_s.sh:69`），於是 manifest 是唯一 gate 且它會 match。情境：第一次 run 用 TARGET=A 生成到一半被中斷（無 meta.json/FINGERPRINT）→ 第二次換 TARGET=B 重跑 data stage → gen_data _resume_复用 A 的舊 rows、只用 B 生成剩下的 rows，最後照樣寫 meta.json 與 B 的 FINGERPRINT。訓練資料被兩個 target 污染，eval 的 `greedy_consistency` 會偏低（部分警訊）但 run 仍完成並回報數字。**Fix:** driver 傳 `--target "$TARGET"` 給 gen_data，並把 target 加進 manifest `keys`，讓 resume 在 target 不同時因 manifest mismatch 硬停。

**2. [MEDIUM] 生成被 SIGKILL/OOM 中斷留下半行 JSONL → 之後每次 resume 與 evaluator 都 crash** — `track_a/spine/gen_data.py:55`（finding `d75cdff6b69bbe7d`）。
rows 用 `out.write(json.dumps(...)+"\n"); out.flush()` append（`:84-86`），SIGKILL/OOM-kill 可能在 write 中途留下不完整的最後一行。resume 在 `:55` 用 `json.loads(line)` 建 done-id set，半行會丟 `JSONDecodeError` 直接中止；同一行也會讓 `eval_accept.py:190` 的 row-id 解析 crash。第二輪的 resume 硬化（manifest + text id）不容忍半行，於是「中斷的生成」可能變成無法自動恢復、需手動刪行。SIGINT 不會撕裂（單一 write 呼叫間才處理訊號），但 pod 上 OOM-kill 是真實情境。**Fix:** resume 時把無法 parse 的尾行截掉（truncate 到最後一個完整換行）再建 done set；eval_accept 對尾行做一致處理。

**3. [LOW，第一輪#9 同類回归] manifest.json 非原子寫入 → 寫入中斷後所有 resume 都 crash** — `track_a/spine/gen_data.py:139`（finding `1aee79021a40051c`）。
第二輪新增的 manifest 用 `mpath.write_text(json.dumps(manifest))`（先 truncate 再寫）。若進程在寫入中途死亡，manifest.json 變半套 JSON；之後每次 resume 在 `:133` 的 `json.loads(mpath.read_text())` 都丟 `JSONDecodeError`，直到手動刪檔。這正是第一輪#9 用 tmp+`os.replace` 修掉的「中斷寫入」類别，現在出現在 gate resume 的 manifest 上。**Fix:** 寫 `manifest.json.tmp` 再 `os.replace`（比照 `eval_accept.py:270-272` 的 summary）。

**4. [LOW，不公平比較] 預設混合 mode sweep 讓 DSpine 有 radix prefix cache、其他 mode 沒有；README 仍宣稱全部關閉** — `track_a/serve_sglang.sh:31`（finding `208033d42278053a`）。
`serve_sglang.sh:31` 對 dspine 保留 radix cache（plugin 必要）、其他 mode 除非 `RADIX_CACHE=1` 否則關閉。`run_baselines.sh:66-87` 對整個 MODES sweep 套用同一個 `RADIX_CACHE`，所以「含 dspine 的預設指令」會給 DSpine prefix-cache prefill 复用（例如 64 個 request 共享的 system prefix），而 ar/mtp/dflash/dspark 沒有 → tok/s 比較不公平。script 内註解有警告，但 `track_a/README.md:19` 仍寫「All modes run with ... the radix prefix cache off」，會誤導操作者。**Fix:** 更正 README（DSpine 需要 radix cache；like-for-like 要設 `RADIX_CACHE=1`）；可讓 run_baselines 在 dspine 與其他 mode 混跑且 `RADIX_CACHE!=1` 時警告或拒跑。

---

## 第一／二輪問題狀態

第一輪 10 項、第二輪 codex 8 項 + Qwen 6 項 + GLM 5 項的修正**大致都有到位**（逐項核對見下）。**唯一確認「仍未修完」的是第二輪 codex#2（resume 換 target）**——即上面發現 1：id 改成 text-hash、manifest 也写了，但 manifest 漏掉 target 身分，且 `DATA_FP` 不 gate resume 路徑。其餘未見殘留。發現 2、3 是第二輪 resume 硬化「新引入/未涵蓋」的邊角，不是舊發現复发。

---

## 各題檢查結果（明確列出查過且正確的）

- **Q1 run_stage_s.sh（fresh + 既有輸出逐 stage trace）：正確，除發現 1 的 gen_data resume 缺口。** 所有 `fp`/`concept_json` 呼叫都是 `x=$(...) || die`（`:62,106,113,132,146,170,171,178`），subshell-die 陷阱已封死；`cached()` 要求 64-hex、空值/不符會 die，不会無聲复用。server 的 EXIT/INT/TERM trap 殺整個 process group（非互動 bash 下 `setsid` 不 fork，`$SERVER` 即 pgid），成功/失敗路徑都停 server 並 `trap -` 清除。quoting 正確（刻意不 quote 的 `${SEEDS:-}`/`$extra` 是 intended split）。failure-as-success 已封：data（rc+meta.json）、capture（`find -name 'data_*.ckpt*'` 計數==rows，`.gz` 已修）、train（DONE 只在 exit 0 後寫）、eval（atomic summary + parse-before-skip）。first-run 不会誤 die（無 marker → rebuild）。`! has data` guard、FEAT_EVAL/FEAT_VAL 路徑、`extra` 改函式參數（smoke 不再洩漏 override）皆正確。
- **Q2 eval_accept.py：正確。** walk 索引無 off-by-one（anchor `first-1..last-1`、`prefix[a-base]`、`n=min(prefix+1, last-a)`、恰好停在 last）；`greedy_segments` 回傳極大 greedy run，與 walk 逐 segment 一致；think-skip（`+2`）在「`\n\n` 為單一 token」前提下正確（第一輪 pod 已驗），且對所有 arm 一致。trimmed `last`/segments 只由 target-side states 算一次，unary/refined 與所有 arm（含 s1_released）完全一致 → 公平。per-domain 用 `rid.split("-")[0]` 正確；`--limit` 在 count check 之後才 slice、rid 用 text 匹配非位置，跳行同時退出總量與 bootstrap pair。`o.hidden_states[...,1:,:]` 必為 (1,A,S,H)（否則 A>1 會 broadcast crash；第一輪 CPU 跑過 → 形狀正確）。
- **Q3 concept.py + patch_specforge.py：正確。** zero-gate bit parity（`tanh(0)*c=0`，`x+0==x`）；causality（window 嚴格 < anchor，aux target stop-grad）；`length` 由 all-zero row 數 right-pad（第二輪已在 pinned checkout 確認 zero padding）；`anchor//2` donor 是 causal/deterministic/batch-independent 的 sound wrong-content control；aux 以 `aux_weight` 加一次、無 leak；`adapter` 與 `next_pred` 的 MLP/輸入/init 完全相同（只差 aux）。concept 注入在 `pre_injection` 之前 → refine/transition_scores 都含 concept（第一輪#7 已修，test_patch 斷言 hidden_states 會變）。**保留項（非發現，無法本機確認）：** 原 loss 是否恰以 `denominator` 正規化（決定 `aux_weight*aux` 權重是否精確）、以及 training 的 position_ids 是否 anchor-first，皆依賴 pinned c0f0a39，本 session 無法讀。
- **Q4 gen_data.py：split/determinism/min-ok/system prompt 正確，resume 有發現 1/2/3。** crc32(text)%100 bucket 互斥、有 disjoint assert、prompt 归屬 intrinsic 不会跨 split 迁移；固定 dataset+args+seed 下 deterministic；任何 request error 或 <97% 即失敗（無 meta.json）；system prompt 與 capture template 一致（第一輪 pod prefix match）。manifest 對「prompt list drift」有效（ids 不同即 mismatch 硬停）——缺口只在 target 身分（發現1）與中斷寫入（發現2/3）。
- **Q5 make_variant.py + fingerprints：正確、公平。** control symlink 同一份 base；concept arm 複製 base 再加 zero-gate tensors（base 不變）；`torch.manual_seed(0)` 在獨立 process 定死 concept init，不擾動訓練 RNG/data order；兩 arm 同 seed/lr/data；per-arm/per-run fingerprint 分开。初始權重/seed/data order 無歧異。
- **Q6 paired_bootstrap.py：正確。** prompt-clustered paired bootstrap、統計量是 pooled accepted length 比值、兩 arm 同一組 resample indices、percentile 索引正確（10k → 250/9750）、zero-denominator guard、dropped/unpaired 計數。
- **Q7 新增項目：** `greedy_segments`+`walk(base)` 正確，**test_eval_walk.py 的 reference 本身也正確**（手推：match loop 停在非 greedy、verifier bonus 要求下一 token greedy、`p=a+n+1`、`t==sum(greedy)`，與 per-segment walk 等價）。prompt_key 匹配對 whitespace/sanitizing 穩健（`" ".join(split())`），對「截斷切到 prompt」會**硬停**（count check 或 `rid is None` SystemExit）而非無聲錯配——是否觸發取決於 SpecForge 截斷方向，本 session 無法確認；唯一性 guard（setdefault/seen_ids）杜絕無聲錯映射。manifest+text-hash id、driver fingerprint/arm/override/`! has data`/FEAT 路徑、`--verify`+versioned marker 皆正確。serve_sglang.sh 的 radix 處理是 intentional（見發現 4 的 fairness/README 缺口）。

**建議：** 上 GPU 前修發現 1（把 TARGET 納入 manifest，這是最可能在長 run 中斷後踩到的無聲污染）；發現 2/3 顺手修（resume 原子性/容錯）；發現 4 改 README 並在混合 sweep 時設 `RADIX_CACHE=1`。預設路徑（fresh run、不中斷、TARGET 不變）我 trace 為正確。
---

## glm

# Stage S 第三輪 QA（唯讀）

已用 `register_finding` 記錄 3 個新問題（review `722bac268c8c0ba5`，decision: needs_verification）。本機（Mac）無 torch，CPU 測試無法在本機重跑；`bash -n` 兩個 driver＋全部 Python 檔 `py_compile` 通過（envelope `e9c027b20052455f`）。測試在第一、二輪已於 pod 通過，上 GPU 前請再跑一次。

## 第一、二輪修復狀態：全部到位，無殘留

逐項驗證：`draft_state_dict` 載入、system prompt＋greedy_segments 修剪＋think-block skip、`anchor//2` donor、val/eval 分集、生成完整性閘（error>0 或 <97% → 無 meta.json＋非零退出）、aux 排除 anchor 0＋real length（零填充已在 pinned checkout 確認）、concept 注入移到 `pre_injection` 之前、每步 fingerprint＋`|| die` 防 subshell、mtime 目錄指紋、`.ckpt*` glob、文字雜湊 id＋manifest、版本化 marker＋`--verify`、S1 指紋含 config、`TARGET` 進訓練、未知 arm 會 die、`! has data` 守門、overrides 改為函式參數、atomic summary＋parse-before-skip、EXIT/INT/TERM trap。**沒有任何一項仍然錯誤。**

## 新發現（依嚴重度）

**1. [MEDIUM] `eval_accept.py:239-244` — prompt 截斷會讓整個 evaluation 中止。** gen_data 允許 16..6000 字元的 prompt；加上 system prompt、空 think block 與最多 512 個 answer tokens，部分對話超過 capture 的 `--max-length 2048`（`run_stage_s.sh:119`）。列對應靠解碼出的 user 文字；若 SpecForge 的截斷落在 prompt 側（左/中截斷，或吃掉 `<|im_start|>user\n` 標記使 split 回傳整段文字），`row_ids.get(prompt_key(user))` 回 None → `SystemExit "cannot match its prompt"`，前面已算完的列全部作廢。不是靜默錯，但會在第一次 GPU run 擋死 eval。**修法：** gen_data 先用 tokenizer 過濾（system+prompt 超過 ~1500 tokens 即丟棄，確保＋512 答案 ≤2048）；eval_accept 加 fallback（用 assistant 開頭文字匹配，截斷列改成 skip＋警告）；並在 pod 確認 SpecForge 截斷方向。

**2. [LOW] `run_stage_s.sh:132-134, 178-179` — train/eval 快取身分不含 SpecForge 套件本身與 driver 內建字面值。** 指紋只涵蓋 4 個被 patch 的檔案＋顯式參數；其餘 SpecForge 程式碼（trainer、collator、mask）與 driver 裡的 `--max-length 2048`、`--batch-size 16` 不在任何指紋內。更新 checkout 或改這些字面值後，DONE/`.fp` 全部照舊比對成功 → 快取的訓練/評估結果被靜默重用（正是第一輪 #8 修掉的類別）；recipe 的 `data.max_length` 也必須與 capture 字面值同步，否則無人察覺。**修法：** 把 `"$SPECFORGE/specforge"` 目錄與 driver 腳本本身加進 CODE_FP／train／evaluate 指紋；max_length 單一來源。

**3. [LOW] `gen_data.py:139` — manifest.json 首次寫入非原子。** 中途被砍會留下半寫檔，之後每次 resume 都在 `json.loads`（`:133`）崩潰，且復原訊息要求刪掉整個目錄、丟棄已寫好的列。**修法：** 比照 `eval_accept.py:270-272` 的 tmp＋`os.replace`。

## 各題檢查結果（查過且正確的明確列出）

- **Q1（driver，fresh＋既有輸出）：** 正確。fresh/快取路徑、trap（註冊於 launch 後、清除於收尾 die 前、成功與失敗都停 server）、`has data`/`! has data` 守門鏈（任何輸入變更會在下游階段被 stale-die 擋下）、所有 `$(fp ...)` 皆 `|| die`、`${c:+--concept "$c"}` 的引號分組正確、smoke/train 的 run 命名與 `rm` 一致、`rm -f "$OUT/smoke_$arm".*` 防舊 marker。皆無 first-run 誤死（保守的 by-design 死法除外）。剩餘缺口即發現 2。
- **Q2（eval_accept）：** walk 索引（`a-base` 有界、`n≥1` 必前進、clip 於 `e`）、greedy_segments、think-skip `+2` 錯位時會被 greedy 檢查自癒（非 greedy 只當 context）、per-domain（`rid.split("-")[0]`）、`--limit`（計數檢查在切片前、配對按文字）皆正確。**trimmed `first/last` 只由 record 決定，unary/refined 與所有 arm 完全一致。** 區段末輪的自願者加碼 token 在線下模擬中保守捨棄，對所有 arm 一致。
- **Q3（concept＋patch）：** 零閘位元同步（`x+0`，torch.equal 對 ±0 亦真）、因果（開閘下的 model-level perturbation 測試；donor `a//2 < a` 且批次無關）、length 零行啟發式（零填充已驗證）、`anchor//2` 是 causal、確定、批次無關的錯誤內容對照組、aux 只以 `aux_weight` 加一次且 stop-gradient、`adapter` 與 `next_pred` 的 MLP 與輸入完全相同（只差 aux）。正確。
- **Q4（gen_data）：** crc32 分桶互斥、manifest 防漂移（id 含 prompt 雜湊，設定＋完整 id 清單生成前寫入、resume 必須相等）、min-ok、system prompt 皆正確。缺口只剩發現 3。
- **Q5（make_variant/指紋）：** control symlink 同一份 base；concept 複製同一 tensors＋零閘模組（seed 0，next_pred/adapter/shuffled 初始化相同）；同 seed/lr/data；每 arm 有自己的 init 指紋，鏈到 run 指紋。公平。
- **Q6（paired_bootstrap）：** prompt 叢集配對重抽、統計量為 pooled accepted length 比值、同一組重抽索引、百分位索引與零分母防護皆正確。
- **Q7（新增項）：** `greedy_segments`/`walk(base)` 正確；**`test_eval_walk.py` 的 reference 本身是對的**（逐行推導：只在 greedy token 起輪、提案上限 block-1、verifier 加碼僅限 greedy 範圍內，與實作組合在語義上等價且為線下設計的正確編碼）；prompt_key 的空白正規化能吸收 strip 類差異、重複 prompt 有 loud 檢查——唯一破口是截斷（發現 1）；FEAT_EVAL/FEAT_VAL 預設父目錄隨 FEAT_TRAIN 建立、evaluate/train 正確取用；`! has data` 守門正確；`--verify`＋版本 marker 邏輯密合（MARK 存在 ⇒ 當前 patch 已套、內容另由 CODE_FP 雜湊）；`serve_sglang.sh:31` dspine 保留 radix cache（其他模式預設關、有文件說明對照開關）。
- **Q8：** 未重提（並在 Q3 依 `check_no_gold_tokens` 引用確認僅讀 anchor token）。
---

## muse

Round three complete. I verified every round-one and round-two fix against the code, the pinned SpecForge commit `c0f0a39` (fetched for the interfaces the local bundle lacks), the real Qwen3-8B tokenizer, and local execution (20,000-trial walk fuzz, fingerprint and shell-guard probes). Nothing from rounds one or two is still wrong. Two new findings below, most severe first.

## Finding 1 (high) — `training.num_epochs: 2` silently caps training below `STEPS`

[track_a/spine/configs/qwen3-8b-spine-offline.yaml:24](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/configs/qwen3-8b-spine-offline.yaml:24), invoked at [run_stage_s.sh:182](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/run_stage_s.sh:182)

SpecForge's `fit()` loops `for epoch in range(self.epoch, self.num_epochs)` with `max_steps` only as an early-stop cap, so a run trains at most `num_epochs × rows / batch_size` steps (2 × 3000 / 2 = 3000 at defaults — exactly the default `STEPS`, correct only by fragile arithmetic). Raising `STEPS` for the long runs (or lowering `TRAIN_PER_DOMAIN`) silently trains fewer steps than the run name, `DONE` fingerprint, and eval summaries claim: e.g. `STEPS=6000` trains 3000 steps, exits 0, and is reported as an `n6000` run, corrupting any comparison across step counts. Nothing parses the final `global_step`, and with `save_interval: 0` only best-step checkpoints exist so the shortfall is unrecoverable from artifacts. **Fix:** make `max_steps` the binding constraint by setting `num_epochs: 1000000` in the recipe (schedules use `total_steps`, which already defaults to `max_steps`, so this is safe), and add a driver guard in `train()` that dies unless `num_epochs × train_rows / batch_size ≥ STEPS`.

## Finding 2 (low) — no prompt token-length guard; one extreme prompt aborts capture after generation

[track_a/spine/gen_data.py:36](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/gen_data.py:36)

`load_prompts` allows prompts up to 6000 chars with no token-length filter (unlike `phase2/gen_distill.py`'s `--max-prompt-tokens`). I verified the full mechanism at `c0f0a39`: the parser truncates keep-first at 2048, then `loss_mask_filter=has_consecutive_supervised_tokens` drops any row left with fewer than 2 adjacent supervised tokens, and the driver's `n == rows` count check then dies — after the expensive SGLang generation, requiring a data wipe and regen (manifest mismatch). If the prompt alone exceeds ~2000 tokens, prompt matching instead fails loudly in eval. Realistic corpora here are English prose (likely ≤1800 tokens), so the trigger is unlikely and every outcome is loud, never silent. **Fix:** filter prompts to `--max-prompt-tokens 1500` in `split_prompts` and record the threshold in `manifest.json`.

## Rounds one and two — all fixes verified correct and complete

Spot checks beyond re-reading: patch anchors are unique at `c0f0a39` (all 6 edits); `DSpineConfig` defaults fill the converted config; `transition_scores`/`select_cached`/`refine` signatures match eval usage; unused sampler slots are 0 (aux exclusion sound); mask context is strictly `< anchor` with isolated blocks; capture shuffles (seed 42), truncates keep-first, prepends the template system prompt, sanitises user text verbatim, persists exactly the keys eval reads, and honors both `.ckpt` extensions; checkpoints strip the `draft_model.` prefix (eval's `removeprefix` is a safe no-op) with `{run}-step{N}/training_state.pt` + `{run}-best` layout and an always-emitted `eval/simulated_acc_len`; the aux is added exactly once and excluded from the `ce_only` prepass; walk/segments/test-reference agree on 20,000 fuzz trials; `cached`/fingerprint/`concept_json`/expansion semantics all probed live; `</think>` skip lands exactly (verified tokenization: `[<think>, \n\n, </think>, \n\n]`, `index+2` skips all four); prompt round-trip is exact for whitespace/unicode/code/5000-char inputs with the real tokenizer (`clean_up_tokenization_spaces=False`).

## Questions checked and found correct

- **Q1 (driver trace):** correct — fresh/existing paths, `|| die` on every fingerprint (64-hex enforced), `concept_json` caller-checks, `! has data` guard, atomic summary + parse-before-skip, EXIT/INT/TERM server cleanup, no quoting/word-splitting bugs, no failure-as-success path. (FEAT parents: train/eval share one parent that is created; capture `mkdir`s the rest.)
- **Q2 (eval walk/trim/accounting):** correct — segments are maximal greedy runs, walk indices exact, trimmed spans identical for both decoders and all arms (target-side only), skips arm-independent, per-domain and `--limit` mapping correct.
- **Q3 (concept + patch):** correct — zero-gate parity, strict causality (mask and module conventions match), `length` sound given verified zero right-padding and used only for the aux window cap, `anchor//2` a causal batch-independent wrong-content control, aux detached and single-counted, `adapter` identical to `next_pred` but for the aux.
- **Q4 (gen_data):** correct except Finding 2 — hash buckets disjoint, manifest blocks drifted resumes, min-ok/errors gate `meta.json`, system prompt matches the template.
- **Q5 (variants/fingerprints):** correct — identical base weights, seed-0 concept init, same seed/data order, per-arm fingerprints.
- **Q6 (bootstrap):** correct — paired prompt-clustered ratio of pooled rates, percentile indices right.
- **Q7 new items:** test reference right (fuzzed); prompt matching robust-or-loud (empirically verified, truncation keep-first); manifest/text-hash ids sound; fingerprint/arm/override/`! has data`/FEAT changes correct; `--verify` + versioned marker close half-patching; `serve_sglang.sh` radix default documented with the equalizer.
