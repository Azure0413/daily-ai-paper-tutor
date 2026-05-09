# ============================================================
# Round 4.5: Aggregator / Summarizer  [NEW]
# 對應 MoA paper 的 aggregator + MAD-summarizer paper 的 history compression
# ============================================================
AGGREGATOR_SYSTEM = """你是 critique aggregator。輸入是一篇文章原稿,以及三份不同角度的 review (數學、概念、格式)。
你的任務是把三份 review 壓縮成一張結構化的「修改清單」,給最終 refiner 用。

嚴格規則:
1. 三份 review 中標記 NO_ISSUES 的,完全略過,不要重述。
2. 不同 review 提到同一個問題就合併成一條。
3. 矛盾處理:若 review 之間衝突 (例如數學要加細節,格式要縮短),
   由你權衡後寫出最終決定,並標註理由 (一句話)。
4. 每一條格式如下:
   [面向][嚴重度] 問題描述 → 修法
   面向 = 數學 / 概念 / 格式
   嚴重度 = HIGH / MED / LOW
5. 排序:HIGH 在前,同嚴重度按「正確性 > 概念 > 格式」排。
6. 整份輸出嚴格控制在 400 字內。簡潔、可執行、無客套。
7. 若三份 review 通通沒問題,只輸出一行:ALL_CLEAN
"""

def aggregator_user(article: str, math_r: str, concept_r: str, format_r: str) -> str:
    # 原稿只截前 1500 字給 aggregator,它不需要看完整內容,只需上下文定位
    return f"""[原稿節錄]
---
{article[:1500]}
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
# Round 5: Final Refiner  (修改:現在收 action_list 而不是三份 raw review)
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
