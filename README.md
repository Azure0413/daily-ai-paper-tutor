# Daily AI Paper Tutor

每天從 NeurIPS / ICLR / ICML / CVPR 等頂會 paper 中挑一個演算法/數學主題,經過多階段 self-correction 產生繁體中文教學,自動推送到 Discord。

---

## Agent Pipeline

參考 **Self-Refine** (Madaan et al., NeurIPS 2023)、**Reflexion** (Shinn et al., NeurIPS 2023)、**Mixture-of-Agents** (Wang et al., 2024)、**MAD with Summarizer** (Smit et al., 2023)、**SID** (Sun et al., 2025) 的設計原則。

```
Round 1a: Picker (gpt-oss-120b + browser_search) → 一行 JSON {title, arxiv_id, venue}
Round 1b: arXiv 驗證 + 去重 (程式,不經 LLM) → 通過才進入寫作
Round 1c: Writer (gpt-oss-120b,不帶工具;拿到真實摘要) → draft
              │ draft
   ┌──────────┼──────────┐
Round 2     Round 3     Round 4
Math Critic  Concept    Format
              │ 3 份 critique
   Round 4.5: Aggregator         ← MoA aggregator + MAD summarizer
   把 critique 壓成可執行 action list,並裁決衝突
              │ action_list
   Round 5: Refiner              ← 整合修正,輸出最終版
              │
           Discord
```

---

### Round 1 — Picker / 驗證 / Writer

**為什麼拆成三步?** 實際跑過 90 天後發現:
1. LLM 用 `browser_search` 挑 paper 時會編出不存在的 paper (標題、作者、會議混搭),而且同一個 paper 會換個說法反覆出現 (歷史檔 90 筆只有約 54 個不重複主題)。
2. `tool_choice="required"` 時 gpt-oss 偶爾直接寫答案不呼叫工具,Groq 回 `400 tool_use_failed`。

所以:
- **1a Picker**:`browser_search` + `tool_choice="required"`,但只輸出一行 JSON (短輸出,失敗成本低);`tool_use_failed` 視為暫時性錯誤自動重試。
- **1b 驗證 (程式)**:用 arXiv API 確認 paper 存在、標題吻合;查無此 paper → 拒絕並換候選。再以 arXiv id 去重 (舊資料退回標題比對)。arXiv 連不上時降級為「未驗證」並在訊息尾端標註。
- **1c Writer**:不帶工具,拿到 arXiv 真實標題 / 作者 / 摘要才寫。若 paper 沒有可教學的數學核心,回 `SKIP` 換下一個候選。

文章尾端的「出處」行由程式用 arXiv metadata 產生,不由 LLM 寫。

---

### Round 2–4 — 三個專門化 Critic

對應 Reflexion 的 Evaluator 拆分。`gpt-oss-120b` 同時但獨立扮演三個角色,每個只專注一個面向:

| Critic | 檢查項目 |
|---|---|
| **Math** | 等式是否成立、是否跳步、偏微分/積分/期望值展開、上下標一致性、維度、機率分布條件 |
| **Concept** | paper 出處正確性、演算法定義是否與原論文一致、動機描述、術語標準化 |
| **Format** | 是否殘留 LaTeX、Unicode 數學符號使用、字數是否符合 Discord 上限、行內排版 |

每個 critic 若無問題則只回傳單行 `NO_ISSUES` 短路,大幅減少下游 token 用量(對應 MAD-with-summarizer 的 history overwrite 策略)。Critique 上限 1500 字,超過則保留開頭並截斷,確保 critique 之間 token 平衡。

**為什麼角色拆分而不是單一 critic?**
單一 critic 容易強化模型自身的確認偏誤 (confirmation bias);三個獨立角色強迫模型從不同維度重新審視,實證上比「同 prompt 跑五次」有效得多 (Reflexion ablation 結果)。

**為什麼用同一個強模型而非多模型混合?**
依 Self-MoA (Li et al., 2025):單一強模型多次採樣 ≥ 不同模型混合 (quality > diversity),且免費層每個 model 有獨立 RPD,集中用一個反而簡單可控。

---

### Round 4.5 — Aggregator (本系統最核心的設計)

對應 Mixture-of-Agents 的 aggregator 層 + MAD-with-Summarizer 的 history compression。此階段把三份原始 critique 壓縮成一張**結構化 action list**:

```
[面向][嚴重度] 問題描述 → 修法
範例:[數學][HIGH] 第 3 步偏微分對 σ 寫成對 σ²,符號不一致 → 統一為 σ²
```

責任:

1. 略過 `NO_ISSUES` 的 review
2. 合併不同 critic 提到的同一問題
3. **裁決矛盾** — 例如 Math 要求補細節 vs Format 要求縮短,由 aggregator 一句話權衡
4. 依嚴重度 (HIGH > MED > LOW) 與面向優先序 (正確性 > 概念 > 格式) 排序
5. 全部 NO_ISSUES → 輸出單行 `ALL_CLEAN`,觸發二段 early exit

**這層解決什麼?** 純文字 multi-agent debate 的 token 使用會二次方累積 (Cross-Modal Memory Compression for MAD, 2026 觀察)。原始的 Round 5 若直接收三份 raw critique,輸入會 ~3500 tokens,撞 free tier TPM。經 aggregator 壓縮後 Round 5 輸入降到 ~1400 tokens,**−60% token**,而且 refiner 拿到的是已排序、已去衝突的清單,修改信號更清晰。

---

### Round 5 — Refiner

接收原稿 + action list (而非三份 raw critique),嚴格依清單修改:HIGH 必改、MED 盡量改、LOW 視情況。不允許新增清單沒提到的內容,避免越改越偏。

---

### Early Exit

對應 SID 的 confidence-based early exit。兩個位置可短路:

1. **Round 4 後**:三個 critic 全部 `NO_ISSUES` → 跳過 aggregator + refiner,直接用 draft
2. **Round 4.5 後**:aggregator 判定 `ALL_CLEAN` → 跳過 refiner

對應 Self-Refine 論文的觀察:過度迭代會導致 quality regression,沒事就不該瞎改。

---

### Continuation 機制

長輸出 (Writer、Refiner) 只在 `finish_reason == "length"` (真的被截斷) 時續寫,用多輪對話格式 (assistant 已寫的部分 + 「請接著寫」),最多 2 次,並自動去除模型重複的重疊段落。模型自己 `stop` 就視為寫完,不會硬續。

> gpt-oss 是 reasoning model,reasoning tokens 也算在 `max_completion_tokens` 內。空輸出 + `finish_reason=length` = 預算被思考吃光,程式會自動加大預算重試。

---

## 穩定性設計

| 問題 | 處理 |
|---|---|
| `400 tool_use_failed` (模型沒呼叫工具) | 視為暫時性錯誤,重試;其他 400 才是永久錯誤 |
| 429 | 依 `retry-after` 等待;>300 秒視為日額度用盡 → 換備援 model |
| 413 (free tier 單次 8K TPM) | 解析 `Limit / Requested`,精準縮減 `max_tokens` |
| 主 model 不可用 | 自動換 `openai/gpt-oss-20b` (可用環境變數 `GROQ_MODEL` / `GROQ_FALLBACK_MODEL` 覆寫) |
| 單一 critic 失敗 | 該份審查視為缺席並在訊息標註,不中斷整條 pipeline |
| Refiner 輸出被截斷 / 太短 | 驗證失敗 → 退回已通過驗證的草稿 |
| 最終文章不合格 (太短 / 缺「一句話總結」) | 不發送,Discord 收到失敗通知 |
| Discord 送出失敗 | raise (不再默默當成功);先送出成功才寫歷史 |
| LLM 輸出 `@everyone` | `allowed_mentions` 全關 |
| 歷史檔毀損 | 明確報錯,不覆寫 |

`python -m pytest -q` 以 mock 驗證上述行為 (不吃 Groq 額度),push / PR 時由 `tests.yml` 自動執行 (與每日排程分開,測試壞掉不會害當天發不出去)。

---

## 模型配置

| 用途 | 模型 | 選用原因 |
|---|---|---|
| Round 1a | `openai/gpt-oss-120b` + `browser_search` | `groq/compound-mini` 已於 2026-09-21 下架,改用 gpt-oss 內建瀏覽工具 |
| Round 1c–5 | `openai/gpt-oss-120b` | Groq 免費層最強 reasoning model |
| 備援 | `openai/gpt-oss-20b` | 獨立的 TPM/TPD 額度 |

皆為 Groq free tier,無需信用卡。

---

## References

- Madaan et al. *Self-Refine: Iterative Refinement with Self-Feedback*. NeurIPS 2023.
- Shinn et al. *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 2023.
- Wang et al. *Mixture-of-Agents Enhances Large Language Model Capabilities*. 2024.
- Smit et al. *Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs*. 2023.
- Li et al. *Rethinking Mixture-of-Agents: Is Mixing Different LLMs Beneficial?* 2025.
- Sun et al. *SID: Multi-LLM Debate Driven by Self Signals*. 2025.
