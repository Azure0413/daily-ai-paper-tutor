import os
import re
import time
from groq import Groq, APIStatusError

client = Groq(api_key=os.environ["GROQ_API_KEY"])

# 內建 web search,Round 1 使用。compound-mini = single tool call、低 token 開銷
MODEL_WITH_SEARCH = "groq/compound-mini"
# 強推理模型,用於 Round 2~5
MODEL_REASONING = "openai/gpt-oss-120b"


def _call_groq_raw(model: str, system: str, user: str,
                   max_tokens: int, temperature: float,
                   max_retries: int = 4) -> tuple[str, str]:
    """底層呼叫,回傳 (text, finish_reason)。
    finish_reason 依 Groq/OpenAI 規格:'stop' (正常結束) / 'length' (撞到 max_tokens) /
    'tool_calls' / 'content_filter' 等。
    """
    last_err = None
    current_max = max_tokens
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_completion_tokens=current_max,
            )
            choice = resp.choices[0]
            text = (choice.message.content or "").strip()
            finish_reason = choice.finish_reason or "unknown"
            return text, finish_reason

        except APIStatusError as e:
            last_err = e
            status = getattr(e, "status_code", None)
            print(f"[Groq] APIStatusError {status}: {e}")
            if status == 413:
                # Request too large / TPM。砍半輸出 + 等 60s 讓 TPM 視窗 reset
                current_max = max(256, current_max // 2)
                print(f"[Groq] 413 -> halve max_tokens to {current_max}, wait 60s")
                time.sleep(60)
                continue
            if status == 429:
                wait = 60 * (attempt + 1)
                print(f"[Groq] 429 -> wait {wait}s")
                time.sleep(wait)
                continue
            time.sleep(2 ** attempt)

        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"[Groq] attempt {attempt+1} failed: {e}, retry in {wait}s")
            time.sleep(wait)

    raise RuntimeError(f"Groq failed after {max_retries} retries: {last_err}")


def call_groq(model: str, system: str, user: str,
              max_tokens: int = 1500,
              temperature: float = 0.4) -> str:
    """單次呼叫,只回傳 text。給短輸出 (critic / aggregator) 用。"""
    text, _ = _call_groq_raw(model, system, user, max_tokens, temperature)
    return text


# 文章是否已經自然結束的判斷字尾(支援中文標點)
_ENDING_PUNCT = ("。", ".", "!", "?", "!", "?", "」", "”", ")", ")")
_ENDING_KEYWORDS = ("一句話總結", "總結:", "總結:")


def looks_complete(text: str) -> bool:
    """粗略判斷一段文章是否寫完。優先用收尾關鍵字,再退回標點判斷。"""
    if not text:
        return False
    last = text.rstrip()[-30:]
    if any(kw in last for kw in _ENDING_KEYWORDS):
        # 出現「一句話總結」字樣,且後面有東西才算寫完
        idx = max(text.rfind(kw) for kw in _ENDING_KEYWORDS if kw in text)
        tail = text[idx:].strip()
        return len(tail) >= 15
    return last.endswith(_ENDING_PUNCT)


def call_groq_complete(model: str, system: str, user: str,
                       max_tokens: int = 1800,
                       temperature: float = 0.4,
                       max_continuations: int = 3) -> str:
    """長輸出用。若 finish_reason=='length' 或結尾不像寫完,
    自動用 continuation prompt 續寫,直到 finish_reason=='stop' 或達到上限。
    
    對應 OpenAI/Groq 官方 streaming continuation pattern:
    https://console.groq.com/docs/api-reference (finish_reason 欄位)
    """
    full = ""
    current_user = user

    for round_num in range(max_continuations + 1):
        text, finish_reason = _call_groq_raw(
            model, system, current_user, max_tokens, temperature
        )
        full = (full + text) if round_num > 0 else text
        print(f"[Groq] call {round_num+1}: finish_reason={finish_reason}, "
              f"chunk={len(text)} chars, total={len(full)} chars")

        # 結束條件:正常 stop 且尾巴看起來完整
        if finish_reason == "stop" and looks_complete(full):
            break
        # 即使 finish_reason='stop',若尾巴明顯不完整,還是要續寫
        if finish_reason != "length" and looks_complete(full):
            break
        if round_num >= max_continuations:
            print(f"[Groq] continuation budget exhausted, returning current output")
            break

        # 續寫 prompt:給模型看尾巴當上下文,要求接著寫
        tail = full[-1500:]
        current_user = f"""{user}

---
你之前已經寫到這裡 (尾巴節錄):
{tail}
---

請從上面的尾巴「無縫接著寫」剩下的部分,直到整篇結束。
嚴格規則:
1. 絕對不要重複任何已經寫過的內容,直接從中斷處的下一個字開始。
2. 不要寫「繼續」、「接著」、「以下接續」之類的開場白。
3. 文章必須以一段「一句話總結:...」結尾,然後就停下。
"""
        print(f"[Groq] -> continuing (call {round_num+2})")

    return full


def truncate_review(text: str, max_chars: int = 1500) -> str:
    """Critic 若回 NO_ISSUES 就直接縮短,否則截斷。對應 MAD-with-summarizer 的 history compression。"""
    text = text.strip()
    if re.search(r"\bNO_ISSUES\b", text) and len(text) < 200:
        return "NO_ISSUES"
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n…(以下已截斷)"
