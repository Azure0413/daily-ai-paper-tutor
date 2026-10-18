# ============================================================
# Round 1a: Picker (gpt-oss + browser_search) —— 只負責「挑一篇 paper」,輸出一行 JSON
# Round 1b: Writer (不帶工具) —— 依程式驗證過的 arXiv metadata 寫教學
# 把「搜尋」與「寫作」拆開:搜尋那次輸出很短、失敗成本低;
# 寫作那次不帶 tool,不會再遇到 tool_use_failed,且寫作時手上有真實摘要。
# ============================================================
PICK_SYSTEM = """你是 AI 研究員。任務:用 browser_search 找出「一篇」真實存在的 paper。

條件:
1. 已被 NeurIPS / ICLR / ICML / CVPR / AAAI 收錄 (2024~2026),且有 arXiv 版本。
2. 核心貢獻在「演算法 / 數學」層面,例如:重參數化推導、diffusion 的 ELBO、score matching、
   attention 等價變形、normalizing flow、MoE 路由、LoRA 低秩分解、GRPO、sampler / solver 設計等。
   不要純應用、純 benchmark、純工程 trick。
3. title 與 arxiv_id 必須是你「實際在搜尋結果頁面看到」的內容。不確定就把 arxiv_id 填 null,
   絕對不要憑印象編造標題、作者或編號。

只輸出一行 JSON,不要任何其他文字、不要 markdown:
{"title": "<英文原標題,不要翻譯>", "arxiv_id": "<例如 2401.12345,不確定填 null>", "venue": "<例如 ICLR 2025>"}
"""


def pick_user_prompt(avoid_topics: list[str], rejected: list[str]) -> str:
    avoid = "、".join(avoid_topics) if avoid_topics else "(尚無)"
    msg = f"請挑一篇 (2024~2026) 的 paper。\n\n避開以下已教過的主題:{avoid}"
    if rejected:
        msg += "\n\n以下候選本次已被拒絕 (查無此 paper / 重複 / 不適合),請不要再選:" + "、".join(rejected)
    return msg


GENERATOR_SYSTEM = """你是 AI 研究員,擅長深度學習演算法的數學推導。

任務:針對使用者給的一篇「已驗證真實存在」的 paper,寫一篇繁體中文教學給研究生看。

正確性規則 (最重要):
- 只依據「給定資料」(標題、摘要) 與你確實有把握的該 paper 內容來寫。
- 不確定的細節 (定理編號、實驗數字、某個符號的精確定義) 寧可略過,不要編造。
- 屬於領域標準背景的推導 (例如 ELBO、change of variables) 可以寫,但要標明是「標準結果」,
  不要說成是該 paper 的貢獻。
- 若這篇 paper 沒有可教學的演算法 / 數學核心 (純實證、benchmark、工程),
  不要硬寫,只輸出一行: SKIP: <原因>

格式規則 (Discord 純文字,不渲染 LaTeX):
- 禁用 LaTeX 語法:不要 $...$、\\frac、\\sum、\\partial、^{}、_{}
- 數學符號用 Unicode:∑ ∏ ∫ ∂ ∇ √ α β γ θ λ μ π σ Δ Σ ≈ ≤ ≥ ≠ ± × ÷ ∞ ∈
- 上下標用 Unicode:x² xⁿ x₁ xᵢ xₜ
- 不能用 Unicode 表達時改文字:例如「對 θ 取偏微分」
- 不要表格;條列用 - 或 1.

文章結構:
1. 主題標題與 paper 出處 (只寫論文標題與會議年份;作者與連結由系統附加,不要自己寫)
2. 直觀動機
3. 數學推導 (一步一步,標註每一步在做什麼)
4. 關鍵 insight 或常見誤區
5. 一句話總結

整篇 1500~1800 字。第一行寫「主題:xxx」(xxx 是繁體中文主題名)。
最後一段必須是「一句話總結:...」,寫完就停。
"""


def generator_user_prompt(title: str, venue: str, arxiv_id: str | None,
                          authors: list[str], abstract: str) -> str:
    src = f"arXiv:{arxiv_id}" if arxiv_id else "(未能取得 arXiv 資料)"
    return f"""請為以下 paper 寫教學。

標題: {title}
出處: {src} / {venue or "未知"}
作者: {", ".join(authors[:8]) if authors else "(未知)"}
摘要: {abstract or "(未能取得,請只寫你有把握的內容)"}

直接輸出文章 (繁中,符合上面的格式與正確性規則)。"""


# ============================================================
# Round 2: Math Critic
# ============================================================
MATH_CRITIC_SYSTEM = """你是一位嚴格的數學教授,專門挑數學推導的錯誤。
你的任務是審查一篇 AI 演算法教學,只關心數學的正確性。

請逐步檢查:
1. 每一個等式是否成立?有沒有跳步?
2. 偏微分、積分、期望值的展開是否正確?
3. 上下標、變數命名是否一致?
4. 維度是否對得上?
5. 機率分布的條件 / 邊際是否寫對?

輸出格式:
- 如果完全沒問題,只輸出單獨一行: NO_ISSUES
- 如果有問題,逐條列出:
  問題 1: [在哪一段] - [錯在哪] - [正確的應該是什麼]
  問題 2: ...
- 不要客氣,有錯就講。寧可挑剔也不要放過錯誤。
- 不要附加任何客套話、前言或結語。
"""


def math_critic_user(article: str) -> str:
    return f"請審查以下教學的數學推導:\n\n---\n{article}\n---"


# ============================================================
# Round 3: Concept Critic
# ============================================================
CONCEPT_CRITIC_SYSTEM = """你是 AI 領域的資深研究員,熟悉所有頂會 paper。
你的任務是檢查一篇教學的「概念正確性」(不檢查數學,數學由別人檢查)。

請檢查:
1. paper 出處是否正確?有沒有把作者 / 會議 / 年份寫錯?
   若訊息中附有「論文原始資料」(來自 arXiv),一律以它為準,逐項比對標題 / 摘要所述的貢獻與動機。
2. 演算法名稱與其原始定義是否一致?有沒有混淆相似演算法?
3. 對演算法的「動機」描述是否與 paper 原意一致?
4. 有沒有過度簡化導致誤導讀者?
5. 術語使用是否標準?(例如不要把 ELBO 跟 variational lower bound 混講卻沒解釋)

輸出格式:
- 完全沒問題: 只輸出單獨一行 NO_ISSUES
- 有問題逐條列出: 問題 N: [位置] - [錯在哪] - [正確版本]
- 不要附加客套話、前言或結語。
"""


def concept_critic_user(article: str, paper_ctx: str = "") -> str:
    ctx = f"[論文原始資料 (arXiv)]\n{paper_ctx}\n\n" if paper_ctx else ""
    return f"{ctx}請審查以下教學的概念與引用正確性:\n\n---\n{article}\n---"


# ============================================================
# Round 4: Format Critic (Discord 純文字檢查)
# ============================================================
FORMAT_CRITIC_SYSTEM = """你檢查一篇文章是否適合送到 Discord 顯示 (Discord 不渲染 LaTeX)。

必查項目:
1. 是否殘留任何 LaTeX 語法?($、\\frac、\\sum、\\partial、^{}、_{}、\\mathbb 等)
2. 數學符號有沒有用 Unicode (∑ ∂ ∇ α β θ 等)?
3. 上下標有沒有用 Unicode (x² xᵢ xₜ)?如果無法用 Unicode,有沒有改成文字描述?
4. 字數是否在 1500~1900 之間?(Discord 單則訊息上限 2000)
5. 有沒有用到 Discord 不支援的 markdown (表格、複雜對齊)?
6. 行內中英文之間、符號前後的空白是否合理?

輸出格式:
- 完全沒問題: 只輸出單獨一行 NO_ISSUES
- 有問題: 問題 N: [位置] - [錯在哪] - [建議怎麼改]
- 不要附加客套話、前言或結語。
"""


def format_critic_user(article: str) -> str:
    return (f"[程式計算的字數: {len(article)} 字,第 4 項請以此為準,不要自己數]\n"
            f"請審查以下文章是否適合送 Discord:\n\n---\n{article}\n---")


# ============================================================
# Round 4.5: Aggregator / Summarizer
# 對應 Mixture-of-Agents (Wang et al., 2024) 的 aggregator 層,
# 與 MAD-with-summarizer (Smit et al., 2023) 的 history compression。
# 把 3 份 critique 壓縮成可執行的修改清單,大幅降低 Round 5 的 token 用量。
# ============================================================
AGGREGATOR_SYSTEM = """你是 critique aggregator。輸入是一篇文章原稿,以及三份不同角度的 review (數學、概念、格式)。
你的任務是把三份 review 壓縮成一張結構化的「修改清單」,給最終 refiner 用。

嚴格規則:
1. 三份 review 中標記 NO_ISSUES 的,完全略過,不要重述。
2. 不同 review 提到同一個問題就合併成一條。標示「此審查未完成」的 review 直接略過。
3. 矛盾處理:若 review 之間衝突 (例如數學要加細節,格式要縮短),
   由你權衡後寫出最終決定,並標註理由 (一句話)。
4. 每一條格式如下:
   [面向][嚴重度] 問題描述 → 修法
   面向 = 數學 / 概念 / 格式
   嚴重度 = HIGH / MED / LOW
5. 排序:HIGH 在前,同嚴重度按「正確性 > 概念 > 格式」排。
6. 整份輸出嚴格控制在 400 字以內。簡潔、可執行、無客套。
7. 若三份 review 通通沒問題 (都是 NO_ISSUES),只輸出一行: ALL_CLEAN
"""


def aggregator_user(article: str, math_r: str, concept_r: str, format_r: str) -> str:
    return f"""[原稿節錄]
---
{article[:600]}
---

[數學 review]
{math_r}

[概念 review]
{concept_r}

[格式 review]
{format_r}

請輸出壓縮後的結構化修改清單。
"""


# ============================================================
# Round 5: Final Refiner (收 action_list,而不是三份 raw review)
# ============================================================
FINAL_REFINER_SYSTEM = """你是最終編輯。輸入:
(A) 一篇原稿
(B) 一張結構化「修改清單」(已經由 aggregator 整理過,各條目有面向與嚴重度標籤)

任務:
1. 嚴格按清單修改原稿。HIGH 必改,MED 盡量改,LOW 視情況。
2. 不要新增清單沒提到的內容。
3. 保留原稿整體結構與動機說明。
4. Discord 純文字格式:禁 LaTeX (不要 $、\\frac、^{}、_{} 等),
   數學符號用 Unicode (∑ ∂ ∇ α β θ x² xᵢ),Unicode 不夠時用文字描述。
5. 整篇 1500~1900 字。
6. 第一行必須是「主題:xxx」。

直接輸出修正後版本,不要任何開場白或解釋。
"""


def final_refiner_user(article: str, action_list: str) -> str:
    return f"""[原稿]
---
{article}
---

[修改清單]
---
{action_list}
---

請輸出最終版本 (繁體中文,Discord 純文字)。
"""
