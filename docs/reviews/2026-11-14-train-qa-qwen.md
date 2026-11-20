# Training pipeline QA: qwen

## NCM-Draft 管線 QA（唯讀，依成本排序）

**1. Gate 1 未強制，plan 模式 gate 顯示 n/a — CONFIRMED**
`run_ncm.sh:62-76` 的 `tok()` 只要 `eval_tokenizer.py` exit 0 就 `touch DONE`，完全不看重建率；而 `eval_tokenizer.py:115-121` 因 `final_level: tokens` 把 gate 判為 plan 模式、`pass=None`（無阈值）。若 tokenizer 重建 <97%，兩條 NCM lane 仍會照跑 30000 steps（數天 GPU 浪費），操作者還可能因「n/a」漏看。
修法：tok 階段讀 `eval_*.json` 的 `token_acc_by_cutoff[-1]`，<0.97（或 0.99）即 `exit 1` 中止。

**2. 腳本在 lane 失敗時仍 exit 0 — CONFIRMED**
`run_ncm.sh:99-100`：`wait` 取得 `r0,r1` 後只 `say "PIPELINE_DONE lanes=$r0,$r1"`，最後一行 `tee` 成功 → 整支腳本 exit 0。drafter 訓練/評估失敗（`rc=1`）不會反映在 exit code，無人值守監控會誤判成功。
修法：結尾加 `[[ $r0 -eq 0 && $r1 -eq 0 ]] || exit 1`。

**3. AMP 下 residual/memory 被提升為 float32，且每次 forward 複製 float32 LM head — CONFIRMED**
`drafter.py:84` RMSNorm 回傳 `(...).to(x.dtype) * self.w`：bf16 × float32 參數 → float32；`group_pos`/`cv` 相加 likewise。故 `mem_norm`/`out_norm` 輸出為 float32，`mem`[512,2560,1024] 記憶體加倍。`drafter.py:311` `F.linear(h, lm_head_weight.to(h.dtype))` 因 h 為 float32，每次把 frozen 248320×hidden head 上拷 ~3-5 GB（autocast 其實會用 bf16，純浪費）。183 GB 應不至 OOM，但數天 run 的記憶體/頻寬浪費可觀。
修法：RMSNorm 改成 `((xf*rsqrt)*self.w.float()).to(x.dtype)`；`token_logits` 先把 h cast 到 `lm_head_weight.dtype`。

**4. tokenizer_ckpt 寫死绝对路徑，與可設定的 ROOT 不一致 — CONFIRMED**
`ncm_b32_k6.yaml:17`、`ncm_b32_k5.yaml:17` 寫死 `/dev/shm/ax/runs/...`，但 `run_ncm.sh:10,18` 的 `RUNS=$ROOT/runs`。預設 ROOT 相符；一旦覆寫 ROOT，drafter 找不到 tokenizer（tok 階段白跑）或誤載舊 `/dev/shm/ax` 的錯 tokenizer。
修法：訓練前由腳本把 `model.tokenizer_ckpt` 用 `--set` 注入 `$RUNS/...`。

**5. distill 階段無完成跳過，resume 仍重載 27B — CONFIRMED**
`run_ncm.sh:28` 只判 `has distill`，無 DONE/存在性守衛；`gen_distill.py:88` 每次都建 vLLM engine。資料已完成時重跑仍會把 27B 載入 GPU 才空轉。
修法：distill 前檢查各 domain `progress.json` 是否已達 `per_domain`，或加 DONE marker。

**6. ROOT/HF_HOME 全放 /dev/shm（tmpfs）— SUSPECTED（環境）**
`run_ncm.sh:10,20`：模型 ~54 GB + 資料 + ckpts 都吃 RAM。pod 的 /dev/shm 若未放大（Docker 預設 64 MB）會在 distill 寫檔時直接失敗。
修法：確認 pod /dev/shm ≥ ~80 GB，或改掛磁碟路徑。

**7. bench 內容可能與訓練集重疊 — SUSPECTED（資料來源）**
`prompts.py:33` bench_math=GSM8K **test**；`gen_distill.py:27` 訓練 math=MetaMathQA（源自 GSM8K）。若 MetaMathQA 含 test 題，bench_math 接受率會偏高。惟 DFlash2 基線用同一批 prompt，gate 3 的「相對」比較仍公平。
修法：對 bench prompt 與訓練 prompt 做去重雜湊檢查。

## checked, correct
- **無資訊洩漏**：`context_view`（drafter.py:314-327）memory 止於 s-2、anchor=seq[s-1] 以 embedding 進入、block 從 s 起；causal+right-pad 安全、SDPA bool mask 確實套用；eval 用 rollout codes 非 gold。最後 committed token 確實可用（cv）。
- **train/eval conditioning 一致**：context_window 2560 ≥ 最長 seq，故 w0=0，兩邊 hidden states 相同；`simulate_cached` 快取因 causal 成立。
- **接受度量正確**：matching prefix +1 bonus（eval_drafter.py:81,149），offset 對齊；survival 用 prefix。
- **NCM recipe**：level 索引/upsampling 正確；corruption 線性均值 0.5 遞減；clean labels（corrective_targets=false）；level_alpha=1 等權；nested dropout 仍更新所有 codebook；dead-code restart。
- **hooks 移除、frozen 27B 包 no_grad、ckpt 不含 27B**（trainable_state_dict 跳 encoder.）。
- **controls 公平**：同 conditioning/seed/資料/decoder 尺寸；bench 僅 val 且不在 train mix。
- 248k head 以 micro_batch=256 控制，B200 可容納。

核心洩漏與 train/eval  mismatch 檢查通過；最高成本風險是 #1/#2 讓失敗分支 silently 跑完數天。
