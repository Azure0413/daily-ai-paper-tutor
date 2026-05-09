import os
import re
import time
from groq import Groq, APIStatusError

client = Groq(api_key=os.environ["GROQ_API_KEY"])

# 內建 web search,Round 1 使用。
# 用 compound-mini 而不是 compound:single tool call、latency 低 3 倍、
# 內部 underlying model 呼叫少,token/TPM 開銷小很多
MODEL_WITH_SEARCH = "groq/compound-mini"
# 強推理模型,用於 Round 2~5 嚴格 critique 與整合
MODEL_REASONING = "openai/gpt-oss-120b"


def call_groq(model: str, system: str, user: str,
              max_tokens: int = 1500,
              temperature: float = 0.4,
              max_retries: int = 4) -> str:
    """呼叫 Groq。
    - 一般錯誤:exponential backoff retry
    - 413 (Request Too Large / TPM 超限):自動把 max_tokens 砍半,並等 60s 讓 TPM bucket reset
    - 429 (Rate Limit):等久一點再試
    
    依 Groq SDK 規格:413/429 raise APIStatusError,可從 e.status_code 判斷。
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
            return resp.choices[0].message.content.strip()

        except APIStatusError as e:
            last_err = e
            status = getattr(e, "status_code", None)
            print(f"[Groq] APIStatusError {status}: {e}")

            if status == 413:
                # 請求太大或 TPM 超限。砍半輸出 budget,等 TPM rolling window 過去 (~60s)
                current_max = max(256, current_max // 2)
                wait = 60
                print(f"[Groq] 413 -> halve max_tokens to {current_max}, wait {wait}s for TPM reset")
                time.sleep(wait)
                continue

            if status == 429:
                # 觸到 RPM/RPD/TPD 限制
                wait = 60 * (attempt + 1)
                print(f"[Groq] 429 -> wait {wait}s")
                time.sleep(wait)
                continue

            # 其他 4xx/5xx
            wait = 2 ** attempt
            time.sleep(wait)

        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"[Groq] attempt {attempt+1} failed: {e}, retry in {wait}s")
            time.sleep(wait)

    raise RuntimeError(f"Groq failed after {max_retries} retries: {last_err}")


def truncate_review(text: str, max_chars: int = 1500) -> str:
    """Critic 若回 NO_ISSUES 就直接縮成短字串;否則截斷到 max_chars。
    對應 MAD-with-summarizer paper 的 history compression 思路:
    overwrite 過長的 review,避免在 aggregation/refine 階段累積冗餘 token。
    """
    text = text.strip()
    if re.search(r"\bNO_ISSUES\b", text) and len(text) < 200:
        return "NO_ISSUES"
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n…(以下已截斷)"
