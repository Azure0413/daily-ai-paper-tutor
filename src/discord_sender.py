import os
import re
import time

import requests

DISCORD_LIMIT = 1900  # Discord 單則上限 2000,留 buffer 給「(i/n)」前綴

# 後備:常見 LaTeX → Unicode 的安全網。「\word」一律加 (?![A-Za-z]) 避免誤傷
# (例如 \in 不該吃掉 \inf、\int 的前綴)
_WORDS = [
    ("sum", "∑"), ("prod", "∏"), ("int", "∫"), ("partial", "∂"), ("nabla", "∇"),
    ("sqrt", "√"), ("alpha", "α"), ("beta", "β"), ("gamma", "γ"), ("delta", "δ"),
    ("epsilon", "ε"), ("theta", "θ"), ("lambda", "λ"), ("mu", "μ"), ("pi", "π"),
    ("sigma", "σ"), ("phi", "φ"), ("psi", "ψ"), ("Delta", "Δ"), ("Sigma", "Σ"),
    ("Omega", "Ω"), ("approx", "≈"), ("leq", "≤"), ("geq", "≥"), ("neq", "≠"),
    ("pm", "±"), ("times", "×"), ("cdot", "·"), ("infty", "∞"), ("in", "∈"),
]
LATEX_FALLBACK = [(r"\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}", r"(\1)/(\2)")]
LATEX_FALLBACK += [(rf"\\{w}(?![A-Za-z])", sym) for w, sym in _WORDS]
LATEX_FALLBACK += [
    (r"\$([^$\n]+)\$", r"\1"),  # 移除 $...$ 包裝
    (r"\\\\", " "), (r"\\,", " "), (r"\\;", " "),
    (r"\\mathbb\{([^}]+)\}", r"\1"),
    (r"\\mathcal\{([^}]+)\}", r"\1"),
    (r"\\text\{([^}]+)\}", r"\1"),
]


def sanitize(text: str) -> str:
    for pattern, repl in LATEX_FALLBACK:
        text = re.sub(pattern, repl, text)
    return text


def split_chunks(text: str, limit: int = DISCORD_LIMIT) -> list[str]:
    """以行為單位切;若切在 ``` 程式碼區塊中間,補上收尾與重開,避免後半篇整個變成 code。"""
    chunks, buf = [], ""
    for line in text.splitlines(keepends=True):
        if len(buf) + len(line) > limit:
            if buf:
                chunks.append(buf)
                buf = ""
            while len(line) > limit:  # 單行就超過,硬切
                chunks.append(line[:limit])
                line = line[limit:]
        buf += line
    if buf:
        chunks.append(buf)

    fixed, reopen = [], False
    for c in chunks:
        if reopen:
            c = "```\n" + c
        reopen = c.count("```") % 2 == 1
        if reopen:
            c = c.rstrip("\n") + "\n```"
        fixed.append(c)
    return fixed


def send_to_discord(text: str, webhook_url: str | None = None) -> None:
    """送出全部分段;任何一段最終失敗就 raise (不能默默當成功,否則會存了歷史卻沒送到)。"""
    url = webhook_url or os.environ["DISCORD_WEBHOOK_URL"]
    chunks = split_chunks(sanitize(text))
    for i, chunk in enumerate(chunks):
        prefix = f"📚 今日 AI 教學 ({i + 1}/{len(chunks)})\n" if len(chunks) > 1 else ""
        # allowed_mentions.parse=[]:就算模型 (或被搜尋內容 prompt-injection) 吐出
        # @everyone / @here 也不會真的標記任何人
        payload = {"content": prefix + chunk, "allowed_mentions": {"parse": []}}
        for attempt in range(4):
            r = requests.post(url, json=payload, timeout=30)
            if r.status_code in (200, 204):
                break
            print(f"Discord send failed: {r.status_code} {r.text[:200]}")
            if r.status_code == 429:
                try:
                    wait = float(r.json().get("retry_after", 2))
                except Exception:
                    wait = 2.0
            else:
                wait = 2 ** attempt
            time.sleep(min(wait, 30) + 0.5)
        else:
            raise RuntimeError(f"Discord 送出失敗 (第 {i + 1}/{len(chunks)} 段)")
        time.sleep(1)  # 避免 rate limit
