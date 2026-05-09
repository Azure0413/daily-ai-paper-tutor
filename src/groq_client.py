import os
import time
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

def call_groq(model: str, system: str, user: str,
              temperature: float = 0.4, max_retries: int = 3) -> str:
    """呼叫 Groq,內建 retry。"""
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_completion_tokens=4096,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"[Groq] attempt {attempt+1} failed: {e}, retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Groq failed after {max_retries} retries: {last_err}")


# 內建 web search 的模型,用於 Round 1 抓 paper
MODEL_WITH_SEARCH = "groq/compound"
# 強推理模型,用於 Round 2~5 嚴格 critique 與整合
MODEL_REASONING = "openai/gpt-oss-120b"
