# Daily AI Paper Tutor

每天從 NeurIPS / ICLR / ICML / CVPR 等頂會 paper 中挑一個演算法/數學主題,經過多階段 self-correction 產生繁體中文教學,自動推送到 Discord。

---

## Agent Pipeline

參考 **Self-Refine** (Madaan et al., NeurIPS 2023)、**Reflexion** (Shinn et al., NeurIPS 2023)、**Mixture-of-Agents** (Wang et al., 2024)、**MAD with Summarizer** (Smit et al., 2023)、**SID** (Sun et al., 2025) 的設計原則。

```
Round 1: Generator (groq/compound-mini + web search)
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

### Round 1 — Generator

使用 `groq/compound-mini` 內建的 web search 工具,搜尋 2024–2026 NeurIPS / ICLR / ICML / CVPR 的演算法主題,產生初版教學草稿。`compound-mini` 比 `compound` 限制為 single tool call、latency 低 3×,token 開銷低很多,適合「搜一次 + 生成」的單純情境。

Prompt 強制限制主題範圍為「演算法 / 數學推導層面」(reparameterization、ELBO、score matching、attention 等價變形、LoRA 低秩分解、GRPO …),排除純應用、純 benchmark、純工程 trick。同時附上 `topics_history.json` 中已教過的主題清單,要求模型避開。

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

長輸出 (Round 1、Round 5) 會偵測 Groq API 回傳的 `finish_reason`:

- `finish_reason == "length"` → 撞到 `max_tokens`,自動續寫
- `finish_reason == "stop"` 但結尾無收尾關鍵字/標點 → 模型誤判已寫完,仍續寫

續寫 prompt 只給尾巴 1500 字當上下文,要求無縫接續、禁止重複、禁止寫「以下接續」開場白。最多 3 輪 continuation,實際通常 1–2 輪即收尾。這對應 OpenAI / Groq 官方的 streaming continuation pattern。

---

## 模型配置

| 用途 | 模型 | 選用原因 |
|---|---|---|
| Round 1 | `groq/compound-mini` | 內建 web search,single tool call,latency 低、token 開銷小 |
| Round 2–5 | `openai/gpt-oss-120b` | Groq 免費層最強 reasoning model |

皆為 Groq free tier,無需信用卡。

---

## References

- Madaan et al. *Self-Refine: Iterative Refinement with Self-Feedback*. NeurIPS 2023.
- Shinn et al. *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 2023.
- Wang et al. *Mixture-of-Agents Enhances Large Language Model Capabilities*. 2024.
- Smit et al. *Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs*. 2023.
- Li et al. *Rethinking Mixture-of-Agents: Is Mixing Different LLMs Beneficial?* 2025.
- Sun et al. *SID: Multi-LLM Debate Driven by Self Signals*. 2025.
