# Stage S code QA, round two — 2026-11-14

Reviewers (read-only): gpt-6.1-sol (codex), Qwen 4 Max preview and GLM 5.3 (ax-code). Gemini 4
Hydrogen produced no output (its headless mode denied file reads twice). Dispositions first, the
three reviews below.

| Source | Finding | Disposition |
|---|---|---|
| codex 1, Qwen 2 | Trim kept non-greedy tokens inside the answer | Fixed: only maximal runs of greedy tokens are scored (`greedy_segments`); other tokens are context. `test_eval_walk.py` compares with a step-by-step reference |
| codex 2, Qwen 3, GLM 5 | Resume keyed on positional ids | Fixed: ids come from the prompt text; `manifest.json` (settings + ids) is written before generation and must match on resume |
| codex 3 | Feature index assumed to equal the JSONL row (SpecForge shuffles with seed 42) | Confirmed in the pinned checkout. Fixed: rows are matched by the prompt text decoded from the features |
| codex 4, Qwen 6 | Fingerprint failure inside `$(...)` did not stop the driver | Fixed: every call is `x=$(fp ...) \|\| die`; `cached` requires a 64-hex digest |
| codex 5 | Directory fingerprint ignored content changes of equal size | Fixed: entries include mtime |
| codex 6, GLM 3 | Re-patching a patched checkout kept old edits; partial patch undetected | Fixed: versioned marker, older versions rejected, `--verify` used by the driver |
| codex 7 | S1 evaluation did not fingerprint its config | Fixed: the effective config is always fingerprinted and passed |
| codex 8 | `TARGET` did not reach training | Fixed: `model.target_model_path="$TARGET"`, in the fingerprint |
| Qwen 1 | Capture count ignored `.ckpt.gz` | Fixed |
| Qwen 4 | Unknown arm trained as control | Fixed: `concept_json` returns 1, caller dies |
| Qwen 5, GLM 1 | Data marker handling (interrupted write; no check when the data stage is not selected) | Fixed: without the data stage the fingerprint must match; with it gen_data re-validates |
| GLM 2 | `TRAIN_OVERRIDES` prefix leaked from smoke into training | Fixed: overrides are a function argument |
| GLM 4 | Length from all-zero rows relies on zero padding | Checked in the pinned checkout: `_slice_and_pad` and the collator pad with zeros. No change |
| GLM note | Could the forward read later block tokens (gold leak in evaluation)? | Checked: without `replace_mask` only the anchor token is read; `test_patch.check_no_gold_tokens` asserts it |

Also changed: held-out test features default to `/workspace` (about 25 KB per token; 900 rows are
23 GB, too much for the RAM disk).


---

## codex

Found eight issues, most severe first. Round-one findings **#2 and #8 remain incompletely fixed**; **#6/#7 can persist when reapplying the patch**.

1. **[P1] Internal non-greedy tokens survive trimming — [eval_accept.py:198](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/eval_accept.py:198).** `keep` ends at the last matching token, rather than the first mismatch. For `[True, True, False, True, True, False]`, it retains five tokens although only two form a greedy continuation. Subsequent anchors then use context the greedy verifier would never produce, invalidating acceptance results. This leaves round-one **#2** unresolved for internal mismatches. **Fix:** truncate at the first mismatch, or reject such rows; retain the same resulting boundary for both decoders.

2. **[P1] Interrupted data generation can resume with different experiment inputs — [gen_data.py:53](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/gen_data.py:53).** Resume checks only ordinal IDs, while the driver writes its fingerprint after completion. After interruption, changing target, generation settings, or prompt counts can reuse old answers under newly assigned IDs and publish them with the new fingerprint. Increasing training count also changes the shuffled prompt lists for val/eval. I reproduced a changed prompt being treated as already answered. **Fix:** persist and validate an input manifest before generating any rows, and validate existing IDs against prompt hashes and generation settings.

3. **[P1] Feature indices are assumed to identify original JSONL rows — [eval_accept.py:211](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/eval_accept.py:211).** The evaluator assigns JSONL line `index` to `data_index`, but SpecForge’s available preprocessing implementation shuffles the dataset with seed 42 before capture. Equal row counts therefore do not establish this mapping: acceptance from a chat prompt can be reported as code or math. **Fix:** preserve original IDs through preprocessing and write an explicit feature-to-ID manifest. This dependency behavior was checked against [upstream preprocessing](https://raw.githubusercontent.com/sgl-project/SpecForge/main/specforge/data/preprocessing.py); that shared helper was absent from the supplied pinned bundle.

4. **[P2] Fingerprint failures do not stop the driver — [run_stage_s.sh:38](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/run_stage_s.sh:38).** Calls such as `f=$(fp ...)` execute `die()` in a subshell, and their failing status is unchecked. Worse, `say()` writes the failure message to stdout, so the captured value is nonempty and passes `cached()`’s guard. I reproduced continuation with a failure message accepted as the fingerprint. **Fix:** send diagnostics to stderr and check every assignment explicitly: `f=$(fp ...) || die ...`.

5. **[P2] Directory fingerprints miss changed feature contents — [fingerprint.py:34](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/fingerprint.py:34).** Directories hash only filenames and sizes. Recapturing different tokens or hidden states with the same tensor shapes commonly preserves checkpoint sizes, so training and evaluation can silently reuse old results. Directory markers are excluded too. Round-one **#8** therefore remains unresolved. **Fix:** hash file contents, or use a validated content-digest manifest generated during capture.

6. **[P2] Reapplying the patch preserves obsolete injections — [patch_specforge.py:108](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/patch_specforge.py:108).** Any occurrence of the unchanged marker skips every edit in that file. Running the current patch on a round-one patched checkout updates `heads/concept.py` but leaves the old injection placement and missing length propagation untouched. The driver’s `cmp` then passes. Thus round-one **#6/#7** can remain active despite a successful patch invocation. **Fix:** version and verify applied replacements; reject or migrate obsolete patch versions.

7. **[P2] Released-arm evaluation omits its config from cache identity — [run_stage_s.sh:134](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/run_stage_s.sh:134).** The config is fingerprinted only when argument four is supplied; S1 omits it. Changing `train_init/config.json`, including `top_k`, can therefore reuse an evaluation produced with different decoding settings. This is another remaining instance of round-one **#8**. **Fix:** always fingerprint the effective config directory, including the default beside the weights.

8. **[P2] `TARGET` overrides do not reach training — [run_stage_s.sh:165](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/run_stage_s.sh:165).** Generation, capture, and evaluation use `$TARGET`, while training keeps the recipe’s hardcoded `Qwen/Qwen3-8B`. Selecting another local checkpoint or revision can train against embeddings/head weights inconsistent with the captured features. **Fix:** pass `model.target_model_path="$TARGET"` and include the effective target identity in the training fingerprint.

Checked and found correct:

- **Q1:** Ordinary stage-command failures propagate; matching cache markers skip work; signal/exit cleanup covers the launched server. Optional JSON/config arguments preserve spaces.
- **Q2:** Teacher-logit alignment, walk indices, verifier bonus, and clipping passed **408 boundary cases**. Both decoders use the same trimmed `last`. Limits/skips retain feature indices; finding 3 concerns their source-ID mapping.
- **Q3:** Module zero-gate parity, causal source windows, right-padded length calculation, unused-anchor exclusion, detached auxiliary targets, single auxiliary-loss addition, and adapter parameter/input matching are correct. The donor is deterministic and causal; it specifically tests stale context.
- **Q4:** Hash buckets prevent cross-split prompt overlap even when candidate lists change. Fixed-input splitting, the 97% threshold/error rule, and the default system prompt are correct.
- **Q5:** Variant construction preserves base tensors and deterministically initializes matching concept modules. Configured training seeds match across arms; full trainer data-order behavior was not runtime-verified.
- **Q6:** The paired prompt bootstrap correctly resamples clusters jointly and computes the ratio of pooled accepted lengths.

Read-only throughout. Local concept/bootstrap checks passed; GPU execution and full pinned SpecForge integration were not run.

---

## qwen

第二輪 QA 完成（唯讀審查）。已用 `register_finding` 記錄 6 個具體問題並以 `review_complete` 結案（decision: request_changes）。以下為報告。

**審查限制（先說明）：** 本機是非 GPU 的 Mac，我嘗試用 `verify_project` 跑 `track_a/test_concept.py` 失敗（此 host 沒有 torch；第一輪記錄這些測試在 pod 上通過），所以無法在本機重跑測試確認。另外 `/private/tmp/.../sf/` 的未 patch SpecForge 參考副本被權限擋下，無法逐行 diff c0f0a39；patch 的正確性是依其內嵌 anchor + `test_patch.py` 的斷言來驗證。以下發現皆為靜態分析，標「條件式」者代表觸發與否取決於我無法在本機確認的執行期事實。

---

## 發現的 bug（依嚴重度排序）

**1. [MEDIUM，若特徵壓縮則會擋死整個 run] `track_a/spine/run_stage_s.sh:113`**（finding `65aa923f52ad20e3`）
driver 用 `find "$2" -name 'data_*.ckpt'` 數 capture 產出的檔案來驗證行數，但這是「精確副檔名」glob，不会匹配 `data_0.ckpt.gz`。而 `eval_accept.py:153` 刻意用 `data_*.ckpt*`、且 `:84` 對 `.gz` 用 `gzip.open`，代表 pinned SpecForge writer 可能輸出壓縮特徵。一旦輸出是 `.ckpt.gz`，`n` 會是 0，`[[ $n == "$rows" ]]` 失敗，`cap` return 1 → `die "capture eval"`，於是第一次 GPU run 會在「其實成功的 capture」上中止。README 明言 capture 從未跑過，所以這條未被測到。
**Fix:** 改成與 evaluator 一致的 `find "$2" -name 'data_*.ckpt*'`；上 GPU 前先在 pinned checkout 確認 `prepare_hidden_states.py` 實際產出的副檔名。

**2. [MEDIUM] `track_a/spine/eval_accept.py:198`**（finding `ec4bc9f9c5743c7d`）
greedy trim 用 `keep = int(greedy.nonzero().max()) + 1`，也就是「保留到最後一個 match」。這只会丟掉純尾部的非 greedy token；只要某個非 greedy token 後面還有 match（例如 SGLang greedy 因 batching 產生的非決定性、或 empty-think skip 的 `+2` offset 差一個 token），該 token 仍会被留在 `[first,last]` 內。但 walk（`:127`）把每個 committed token 都當成 verifier 的 argmax 無條件接受，於是 accept_len 被膨脹——`greedy_consistency` 只会報告 <1，不会修正這個數字。註解宣稱的「drop a non-greedy tail」只有在所有 mismatch 都在尾部時才成立。
**Fix:** 改成保留「開頭連續 greedy run」：`keep = int((~greedy).nonzero().min()) if (~greedy).any() else greedy.numel()`，保證被評估的 span 對每個 arm 都是完全 greedy。

**3. [MEDIUM] `track_a/spine/gen_data.py:54`**（finding `38577f917a48148f`）
resume 用「id 是否已存在」跳過，但 id 是 prompt 在 shuffle+hash-split 後清單中的「位置」(`f"{d}-{i}"`，`:114-116`)。清單只有在 HF dataset、`need`、seed 都不變時才穩定；`DATA_FP` 只 fingerprint 腳本與 per-domain 數量，不含 dataset 內容/版本。若初次生成與 resume 之間 streaming dataset 漂移（或直接以不同數量重跑進同一個 `--out`），位置 i 會對應到不同 prompt：新 prompt 因 id「已完成」被跳過，且同一 prompt 可能落到與已寫入 rows 不同的 split，無聲污染 held-out 集。
**Fix:** 讓 id 由 prompt 文字決定（如 `f"{d}-{crc32(text):08x}"`）而非位置；並把解析後的 prompt 清單（或其 hash）存進 meta.json，resume 時不一致就拒絕。split 指派也用同一個 text hash。

**4. [MEDIUM] `track_a/spine/run_stage_s.sh:155`**（finding `e38f636ceb869380`）
`c=$(concept_json "$arm")` 在 command-substitution 子 shell 執行，`concept_json` 的 `*) die "unknown arm"`（`:54`）只会結束該子 shell，主 script 繼續、`c` 變空字串。空 `c` 正好就是 control 情境，`${c:+--concept "$c"}` 會省略 `--concept`，`make_variant` 建出 control warm-start。所以 `ARMS` 打錯字（如 `nextpred`）會「以該 concept arm 之名訓練並回報一個 control run」——是無聲的錯誤實驗，而非預期的硬停。這正是 Q1 問的 subshell-die 陷阱的具體實例。
**Fix:** 在父 shell 先驗證 arm（`case "$arm" in control|last_chunk|last_query|next_pred|adapter|shuffled) ;; *) die ...;; esac`），或讓 `concept_json` 回傳非零並用 `c=$(concept_json "$arm") || die "unknown arm $arm"`。

**5. [LOW] `track_a/spine/run_stage_s.sh:61`**（finding `58baf28fd2339100`）
gen_data 成功時寫 meta.json，driver 之後才寫 FINGERPRINT。若 driver 在兩者之間被中斷，資料其實完整，但下次 run 會先撞 `:61` 的 guard 而 die，輪不到 `cached` 去重建，於是管線卡死、需手動刪 meta.json。此 guard 无法區分「舊版無 fingerprint 資料」與「fingerprint 寫入被中斷」，訊息也沒說要刪什麼。
**Fix:** 把 FINGERPRINT 與 meta.json 一起原子寫入；或只在「資料要被重用」時才強制 guard——`has data` 且 fingerprint 不存在時應 fall through 去重建而非 die。

**6. [LOW] `track_a/spine/run_stage_s.sh:156`**（finding `b2fe55c2dd397d84`）
不同於 data/capture/evaluate 路徑，`train()` 的 init fingerprint `f=$(fp ...)` 沒有經過 `cached()` 的「空 fingerprint 即 die」檢查。若 fingerprint.py 在子 shell 失敗，`f` 為空；mismatch 判斷會重建 variant，然後 `echo "$f" > FINGERPRINT` 寫入空 marker，run fingerprint 再用空字串算出、訓練照常進行。下次 invocation 會自癒（真 hash ≠ 空）但會無聲重建並留下假 marker。需要 fp 本身失敗才觸發，機率低；因 Q1 明確問 subshell fingerprint 而列出。
**Fix:** 比照其他 call site 用 `f=$(fp ...) || die "init fingerprint"`（在父 shell die），或寫 marker 前先拒絕空 `f`。

---

## 第一輪問題狀態

第一輪 10 項的修正「都有到位」，本輪未見任何一項殘留：`draft_state_dict` 載入、system prompt + greedy trim、causal `anchor//2` donor、val/eval 分集、生成完整性 gate、aux 排除 anchor 0 + 傳 real length、concept 注入移到 `pre_injection` 之前、每步 fingerprint、atomic summary、server EXIT/INT/TERM trap。上面 6 項是「新」發現，不是第一輪复发。

## 各題檢查結果（明確列出查過且正確的）

- **Q1 run_stage_s.sh：** 有 bug（發現 1、4、5、6）。正確的部分：server 的 EXIT/INT/TERM trap 會殺整個 process group；`cached()` 對 stale fingerprint 會 die（不会無聲重用）；fresh/既有輸出的_stage 跳過邏輯正確；train/val/eval 分集；主要路徑都有 quoting。subshell-die 陷阱在 data/capture/evaluate 的 fp 路徑会被 `cached()` 的空值檢查擋下，只有 concept_json（發現4）與 train init fp（發現6）漏接。
- **Q2 eval_accept.py：** 有 bug（發現2）。正確：walk 索引（前进 prefix+1、在 last 截斷，無 off-by-one）；row-id 用「檔名索引」映射，所以 `--limit` 或跳行不会錯位（且有 `len(files)==len(row_ids)` guard）；per-domain 用 `rid.split("-")[0]` 正確；trimmed `last` 只由 arm-independent 的 target states 算一次，故 unary/refined 與所有 arm 完全一致。
- **Q3 concept.py + patch_specforge.py：** 全部正確。zero-gate bit parity（`tanh(0)*c=0`）；causality（concept window 嚴格 < anchor，aux target stop-grad）；`length` 由 all-zero rows 數 right-pad 的真實位置、只用於 aux future-window 截斷，正確；`anchor//2` donor 是 causal、deterministic、batch-independent 的 sound wrong-content control；aux loss 以 `aux_weight` 加一次（無 double count）、stop-gradient（無 leak）；`adapter` 與 `next_pred` 的 MLP 與輸入完全相同（只差 aux）。限制：靠靜態閱讀 + 第一輪 pod 測試通過，本機無法跑 `test_patch.py`。
- **Q4 gen_data.py：** 有 bug（發現3）。正確：hash-bucket split（crc32%100、互斥、有 assert 無重疊）；min-ok（任何 request error 或 <97% 即失敗）；system prompt 與 capture template 一致（第一輪已在 pod 做 prefix match）。determinism 在固定 dataset+args+seed 下成立；弱點只在 drift 下的 resume（發現3）。
- **Q5 make_variant.py + fingerprints：** 正確、公平。control 直接 symlink 同一份 base weights；concept 複製 base 再加 zero-gated concept tensors（base 不變）；兩 arm 同 seed/lr/data；concept init 在訓練前以 seed 0 定死，不会擾動訓練 RNG/data order；per-arm fingerprint 分开各 run。初始权重/seed/data order 無歧異。
- **Q6 paired_bootstrap.py：** 正確。prompt-clustered paired bootstrap、統計量是 pooled accepted length 的比值、兩 arm 用同一組 resample prompt indices、percentile CI。無 bug。

**建議：** 上 GPU 前至少修發現 1（先確認特徵副檔名）與發現 4（arm 驗證），並考慮發現 2、3；5、6 屬低機率但顺手可修。
---

## glm

Shell execution and scratchpad reads are both denied in this session, so this review is close-reading only; SpecForge-internal claims rest on the patch anchors and the two CPU test files (which I read in full). Report follows.

---

# Stage S QA — round two (read-only, 2026-11-14)

Verification limits: no shell and no access to the pinned SpecForge checkout (`/private/tmp/.../scratchpad/sf/` reads were denied), so statements about SpecForge internals are inferred from `patch_specforge.py`'s anchors plus `test_patch.py`/`test_concept.py`, exactly like round one's limitation note. The CPU tests were not re-executed in this session.

## Findings (most severe first)

**1. Stale data is silently reused whenever the `data` stage is not selected — run_stage_s.sh:59-62.**
`DATA_FP` is validated only inside `if has data && ! cached "$W/data/FINGERPRINT" "$DATA_FP"` (line 62); line 61 catches only the *missing*-marker case. Trace: run once with `STAGES="data capture s1 smoke" TRAIN_PER_DOMAIN=1000`, later run `STAGES="capture" TRAIN_PER_DOMAIN=500` (or edit `gen_data.py`/`gen_distill.py`, or change `MAX_TOKENS`/`TARGET`). `cached()` never runs, no `die` fires, and capture/s1/train/eval all proceed on data that no longer matches the requested inputs — precisely the stale-reuse class round-one #8 was meant to close; every other cached step (capture, init, train, eval) *does* validate its marker unconditionally when consumed. Fix: make the check unconditional, e.g. replace lines 61-62 with
```bash
[[ ! -f $W/data/meta.json ]] || cached "$W/data/FINGERPRINT" "$DATA_FP"   # dies on missing marker or mismatch
if has data && ! cached "$W/data/FINGERPRINT" "$DATA_FP"; then
```
(`cached` returning 0 on match preserves today's behavior for the `has data` rebuild path).

**2. `TRAIN_OVERRIDES` leaks from the smoke stage into the train stage when both run in one invocation — run_stage_s.sh:176, consumed at 162 and 167.**
`TRAIN_OVERRIDES="training.eval_interval=20 training.log_interval=5" train ...` uses a variable-prefixed function call; in bash (POSIX-required behavior, unlike external commands) such assignments persist in the shell after the function returns. With `STAGES="smoke train"` in a single invocation (nothing forbids it; both blocks run in the same process), every real training run after the smoke loop silently inherits `eval_interval=20`, which changes the cadence of `*-best` checkpoint selection — a materially different selection schedule from the recipe's 250 — and `log_interval=5`. Run names (`arm_lr_seed_steps`) do not record it, so the sweep table would look normal (fingerprints do include the string, so no stale reuse, but the runs are trained with unintended hyperparameters). Fix: pass overrides as a fifth argument to `train()` (e.g. `train "$arm" 3e-5 9 20 "training.eval_interval=20 training.log_interval=5"`, with `local ov=${5:-${TRAIN_OVERRIDES:-}}` used in the fingerprint and the `specforge train` line), or `unset TRAIN_OVERRIDES` immediately after the smoke loop. (I could not run a shell to demonstrate this; it is documented bash/POSIX behavior — verify with `f(){ :; }; V=1 f; echo ${V-unset}` → prints `1`.)

**3. The driver's patched-checkout guard covers only `heads/concept.py`, so a half-patched SpecForge runs silently — run_stage_s.sh:124-127; patch_specforge.py:105-113, 119-120.**
`cmp -s track_a/concept.py $SPECFORGE/.../heads/concept.py` proves `concept.py` is current but says nothing about the three *edited* files. `patch_specforge.py` skips any file already containing the MARK (line 108) and writes the pending files sequentially (119-120), so a checkout where `dspine.py` is patched but `algorithms/dspine/model.py` is not (manual revert of one file, partial application on a mixed-lineage tree) passes the `cmp`, passes `CODE_FP` hashing (a missing file is hashed as a path *string* by `fingerprint.py:36-37`, never an error), and trains fine — but `next_pred` and `shuffled` lose their auxiliary objective and `concept/*` metrics with **no error anywhere**: a silently different method reported under the same arm name. (An unpatched `dspine.py` or legacy codec *is* loud — the conditioner is never built / `_check_fields` rejects `concept_config`.) Fix: after the `cmp`, also require the MARK in the other two files, e.g. `grep -q '# [concept-spine]' "$SPECFORGE/specforge/algorithms/dspine/model.py" "$SPECFORGE/specforge/config/draft/codec/legacy/dspine.py" || die "SpecForge checkout not fully patched; re-run patch_specforge.py on a clean checkout"` — or have `patch_specforge.py` write a manifest of patched-file hashes that the driver verifies.

**4. The aux-target length is a zero-row heuristic whose padding assumption is unverified — patch_specforge.py:52-54 (`length = (target_hidden != 0).any(-1).sum(1)`), concept.py:111-120.**
`length` is derived by counting nonzero rows of `target_hidden` instead of plumbing the real sequence length (round-one #6's fix as implemented). It is correct iff SpecForge's offline collate pads `hidden_states`/`target_hidden` with exact zeros. If padding is anything else, `length == T` for every row, and for anchors near the answer's end the `next_pred` auxiliary target window `[end, min(end+chunk, length))` (concept.py:111-115) extends past the real tokens into padding; `has = (n > 0) & (end > 0)` does not catch it, so the MLP is silently regressed toward padding content for exactly the anchors that matter (end-of-answer blocks). The concept *inputs* are unaffected (sampler anchors stay inside the supervised span), so this contaminates only the aux loss of `next_pred`/`shuffled`. Fix: verify once on the first GPU run (log `(target_hidden != 0).any(-1).sum(1)` next to the real lengths from `input_ids != pad` in the trainer) or extend the patch minimally: the algorithm wrapper has the batch, so set `self.draft_model.concept_length = (input_ids != pad_token_id).sum(1)` there and read it in the forward instead of the heuristic.

**5. `gen_data` resume keys on positional ids over an unpinned streaming dataset — gen_data.py:52-54, 114-116; phase2/gen_distill.py:52.**
`split_prompts` is deterministic given the prompt list, but the list comes from `load_dataset(..., streaming=True).shuffle(seed, buffer_size=20_000)` with **no pinned revision**. If the hub dataset's `main` moves (or `load_prompts`'s filtering changes) between a failed attempt and its resume, ids `{domain}-{i}` enumerate the *new* list while the jsonl keeps rows from the old one: resume skips ids already present, so `code-17` silently stays the old prompt and the new `code-17` is never generated; `written >= min_ok * len(rows)` can pass on the union of two different selections. Cross-split contamination is impossible (crc32 bucketing is per-prompt-text, and main() asserts disjointness), so the damage is silent id/prompt drift and extra rows, not split leakage. Realistic because `data` can legitimately need a second attempt (request errors → no `meta.json`). Fix: write a sidecar manifest of the selected `(id, sha256(prompt))` pairs before generating and, on resume, verify the recomputed selection matches it (abort on drift); minimally, pass an explicit `revision=` to `load_dataset` and record it in `meta.json`.

## Questions checked and found correct

- **Q1 (driver trace, fresh + existing outputs):** Correct after findings 1-3 are carved out. The subshell-fingerprint concern is contained: every `fp` result that guards a cache flows into `cached()`, which dies on an empty string (data:62 via `has data` — see finding 1 for the one gap; capture:100,107; init:157; eval:135), and `CODE_FP`'s empty-string hole needs `fingerprint.py` itself to crash, which the `cmp` at 126-127 mostly forecloses (see finding 3). Quoting is sound everywhere (intentionally unquoted splits: `${TRAIN_OVERRIDES:-}`, `${SEEDS:-}`); `cached`'s marker compare tolerates the trailing newline. Failure-as-success paths checked and closed: data (`rc` + `meta.json`, gen_data exits nonzero when `error>0` or rows < 97%), capture (file-count == rows at 113-114), train (`DONE` written only after a zero exit; dir-without-DONE refused), eval (marker written after the run; atomic summary + JSON-parse gate before skipping). Traps: EXIT/INT/TERM registered right after launch, group kill via the `setsid` session works because a non-interactive `bash script.sh` leaves the launcher a non-group-leader (so `setsid` execs without forking and `$SERVER` is the new pgid); traps are cleared before the post-run `die`, and the server is stopped on both success and failure paths. First-run deaths: none found beyond the conservative by-design ones (interrupted run dir, data-without-fingerprint).
- **Q2 (eval_accept):** Walk arithmetic re-derived and correct: anchors `first-1 .. last-1` index `prefix[a-(first-1)]` in bounds; a round commits `matches+1` clipped to `last-a`, so `a` never overshoots and the loop ends exactly at `last`; `surv[:min(n,S)]` is safe since `n <= S`. The greedy trim (`keep`, `last = first+keep-1`) is computed from target-side data only, before any forward, so it is identical for unary and refined and for every arm including `s1_released` — a fair, shared denominator. Think-block skip: sound given `"\n\n"` is one token (pod prefix check claimed in round one); a literal `</think>` inside the first eight answer tokens would mis-skip, but identically for all arms. Per-domain accounting pools `tokens/rounds` per id prefix (ids are `code|math|chat`-prefixed in all three splits, `v`/`e` infixes don't disturb `split("-")[0]`). Row-id mapping is correct with `--limit` (the count check at 161 runs before the slice; `rid = row_ids[global file index]`, not the sliced position) and skipped rows drop out of both totals and bootstrap pairing. Any shard-restart naming from `prepare_hidden_states` would fail the count checks loudly, not silently.
- **Q3 (concept + patch):** Zero gates give bit parity trivially (`tanh(0)·c = 0`, `x + 0 == x` in IEEE, no NaN source), and `test_patch.check_parity` asserts `torch.equal` on `hidden_states`, `pre_injection`, `layer_features`; the control arm builds no module at all, so it runs the original code path. Causality is proven at model level (`check_causality` perturbs every target state at/after each anchor and requires that anchor's block output unchanged) and module level (`test_concept.check_causality`, both inclusive settings). Anchor 0 / unused slots are excluded from the aux (`end > 0` mask), the aux target is stop-gradient (`test_concept.check_aux_and_grads` asserts no gradient into future positions), and it is added to the objective exactly once per forward, only in training mode, with `adapter`/`last_chunk` returning `None` aux. `adapter` constructs the identical MLP (`concept.py:69-71`) over the identical inputs (`:106-109`) as `next_pred`, differing only in the aux — a sound capacity control. The `anchor // 2` donor is always committed past, deterministic and batch-independent (`check_shuffle`), and the shuffled arm keeps the same aux weight and seed-identical init as `next_pred`, making it a fair wrong-content control. Residuals: finding 4 (length heuristic) and one item I could not re-verify against `c0f0a39`: that the original loss normalizes by `denominator` (so that `aux_weight * concept_aux * denominator` contributes exactly `aux_weight * aux`); the patch's own `concept/aux` metric weighting is consistent with it.
- **Q4 (gen_data):** Split logic correct: crc32(text)%100 buckets with `<20 / 20-29 / rest`, disjoint id namespaces, and an explicit cross-split assert; a prompt's bucket is intrinsic to its text, so no prompt can migrate between splits across re-runs. Resume (by id) plus min-ok (nonzero exit, no `meta.json`) work as round-one intended. System prompt is sent explicitly and defaults to the template's claimed constant; a mismatch is not machine-checked before the expensive capture, but it surfaces afterward as depressed `greedy_consistency` in every summary. The determinism gap is finding 5.
- **Q5 (make_variant / fingerprints):** No divergence path found: init fingerprints include the arm's concept JSON plus `train_init` weights/config, so an arm's variant rebuilds when its inputs change and run fingerprints then go stale-by-design; base weights are the same file (control symlinks, concept arms copy the same tensors and add zero-gate modules whose RNG is fixed at `torch.manual_seed(0)` — identical across `next_pred`/`adapter`/`shuffled`); training seed and data order come from the same recipe fields for every arm; capture uses one `train_init` for all arms.
- **Q6 (paired_bootstrap):** Correct. The statistic is the pooled ratio `sum(tokens)/sum(rounds)` per arm; prompts are resampled jointly (paired, cluster = prompt), the gain is the ratio of the two pooled ratios on each resample, percentiles are indexed correctly on the sorted bootstrap (`boots[250]`, `boots[9750]` for 10k), zero-denominator guards exist, and dropped/unpaired ids are counted. This is the right clustered paired bootstrap for a ratio of pooled rates.

## Round one, revisited

All ten fixes are present in the code as the dispositions claim (checkpoint loading; system prompt + trim + think skip; `anchor // 2` donor; val/eval split; failure → no `meta.json` + nonzero exit; anchor-0 exclusion + lengths — implemented as the heuristic in finding 4; concept before `pre_injection`; fingerprints everywhere — with the data-stage gap in finding 1; atomic summary + parse-before-skip; EXIT/INT/TERM traps). Nothing from round one is confirmed *still wrong*. Two honesty notes: (a) per README:52-55 the whole path — data generation, feature capture, training, accepted-length evaluation on real converted weights — has still never executed, so the converter's serving→training tensor mapping (`convert_dspine.py:41-64`) remains verified only by round trip and shape acceptance, not by behavior; the cheap canary on the GPU run is that `s1_released` accepted length lands well below the block size and near DSpine-b16's published acceptance (a layout or leakage bug — e.g. if `DSpineDraftModel.forward` consumed `block_token_ids` computationally rather than as labels, which `check_causality` does not rule out since it never varies that argument — would push it toward saturation for *every* arm); (b) `capture_hf`'s interface with `prepare_hidden_states.py` (batch iteration, `rows_*/data_*.ckpt` global indexing, zero padding in collation) is likewise still unexercised; all failure modes I could construct there are loud crashes or visible in `greedy_consistency`, except finding 4's aux-target case.