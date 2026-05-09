import os
import re
import time
import requests

WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
DISCORD_LIMIT = 1900  # 留 100 字 buffer

# 後備:常見 LaTeX → Unicode 的安全網
LATEX_FALLBACK = [
    (r"\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}", r"(\1)/(\2)"),
    (r"\\sum", "∑"), (r"\\prod", "∏"), (r"\\int", "∫"),
    (r"\\partial", "∂"), (r"\\nabla", "∇"), (r"\\sqrt", "√"),
    (r"\\alpha", "α"), (r"\\beta", "β"), (r"\\gamma", "γ"),
    (r"\\delta", "δ"), (r"\\epsilon", "ε"), (r"\\theta", "θ"),
    (r"\\lambda", "λ"), (r"\\mu", "μ"), (r"\\pi", "π"),
    (r"\\sigma", "σ"), (r"\\phi", "φ"), (r"\\psi", "ψ"),
    (r"\\Delta", "Δ"), (r"\\Sigma", "Σ"), (r"\\Omega", "Ω"),
    (r"\\approx", "≈"), (r"\\leq", "≤"), (r"\\geq", "≥"),
    (r"\\neq", "≠"), (r"\\pm", "±"), (r"\\times", "×"),
    (r"\\cdot", "·"), (r"\\infty", "∞"), (r"\\in", "∈"),
    (r"\$([^$]+)\$", r"\1"),  # 移除 $...$ 包裝
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
    """以段落為單位切,避免在句中切斷。"""
    chunks = []
    buf = ""
    for line in text.splitlines(keepends=True):
        if len(buf) + len(line) > limit:
            if buf:
                chunks.append(buf)
                buf = ""
            # 單行就超過,硬切
            while len(line) > limit:
                chunks.append(line[:limit])
                line = line[limit:]
        buf += line
    if buf:
        chunks.append(buf)
    return chunks

def send_to_discord(text: str) -> None:
    text = sanitize(text)
    chunks = split_chunks(text)
    for i, chunk in enumerate(chunks):
        prefix = f"📚 今日 AI 教學 ({i+1}/{len(chunks)})\n" if len(chunks) > 1 else ""
        payload = {"content": prefix + chunk}
        for attempt in range(3):
            r = requests.post(WEBHOOK_URL, json=payload, timeout=30)
            if r.status_code in (200, 204):
                break
            print(f"Discord send failed: {r.status_code} {r.text}")
            time.sleep(2 ** attempt)
        time.sleep(1)  # 避免 rate limit
