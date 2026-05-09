import os
import re
import time
from groq import Groq, BadRequestError

client = Groq(api_key=os.environ["GROQ_API_KEY"])

# 內建 web search 的模型,用於 Round 1 抓 paper
MODEL_WITH_SEARCH = "groq/compound"
# 強推理模型,用於 Round 2~5 嚴格 critique 與整合
MODEL_REASONING = "openai/gpt-oss-120b"


def call_groq(model: str, system: str, user: str,
              max_tokens: int = 1500,
              temperature: float = 0.4,
              max_retries: int = 4) -> str:
    """呼叫 Groq。
    - 一般錯誤:exponential backoff retry
    - 413 (request_too_large):自動把 max_tokens 砍半重試,避開 TPM 上限
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

        except BadRequestError as e:
            # 413 / request_too_large:把輸出長度砍半再試
            msg = str(e)
            last_err = e
            if "413" in msg or "request_too_large" in msg or "too large" in msg.lower():
                current_max = max(256, current_max // 2)
                print(f"[Groq] 413 too large, halving max_tokens -> {current_max}")
                time.sleep(2)
                continue
            raise

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
    # 保留開頭(通常是最重要的問題)
    return text[:max_chars] + "\n…(以下已截斷)"
