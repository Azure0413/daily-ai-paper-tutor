"""Groq API 薄封裝:重試 / 換 model / 續寫 / 輸出驗證。

設計重點 (對應實際踩過的坑):
- tool_choice="required" 時,gpt-oss 偶爾會直接寫答案而不呼叫 tool,
  Groq 回 400 `tool_use_failed`。這是「模型行為的隨機失敗」,重試通常就會好,
  所以不能像其他 400 一樣當成永久錯誤。
- gpt-oss 是 reasoning model,reasoning tokens 也算在 max_completion_tokens 裡。
  預算太小 → content 為空且 finish_reason="length"。空輸出一律視為失敗並加大預算重試。
- Free tier 是 8K TPM(單次 request 超過就 413,累積超過就 429)。
  413 依訊息裡的 Limit / Requested 精準縮減 max_tokens;429 依 retry-after 等待。
- 續寫只在 finish_reason=="length" 時才做;模型自己 stop 就是寫完了,不能硬續。
"""
import os
import random
import re
import time

from groq import APIConnectionError, APIStatusError, APITimeoutError, Groq

PRIMARY_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
# 主 model 額度用盡 / 下架 / 反覆失敗時的備援 (獨立的 TPM/TPD 額度)
FALLBACK_MODEL = os.environ.get("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")

MODEL_REASONING = PRIMARY_MODEL
MODEL_WITH_SEARCH = PRIMARY_MODEL  # gpt-oss 系列才支援 browser_search
SEARCH_TOOLS = [{"type": "browser_search"}]
SEARCH_TOOL_CHOICE = "required"

_MIN_TOKENS = 400            # 413 縮減的下限,再小就沒意義
_TOKEN_CEILING = 6000        # 空輸出時加大預算的上限 (+ input 仍須 < 8K TPM)
_QUOTA_WAIT_LIMIT = 300      # retry-after 超過這個秒數 = 日額度用盡,等也沒用


class GroqError(RuntimeError):
    """這個 model 這次不行了 (重試用盡 / 額度用盡 / 下架)。可換備援 model。"""


class GroqPermanentError(GroqError):
    """換 model 也沒用 (金鑰錯誤、請求格式錯誤)。直接中止。"""


_client = None


def get_client() -> Groq:
    """延遲建立,避免 import 階段就因缺少環境變數而在錯誤通知機制之外崩潰。"""
    global _client
    if _client is None:
        # max_retries=0:重試全部由本檔控制,避免 SDK 內建重試與我們的疊加
        _client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=180.0, max_retries=0)
    return _client


def _error_code(e: APIStatusError) -> str:
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict) and inner.get("code"):
            return str(inner["code"])
    m = re.search(r"""['"]code['"]:\s*['"]([^'"]+)['"]""", str(e))
    return m.group(1) if m else ""


def _retry_after(e: APIStatusError):
    try:
        v = e.response.headers.get("retry-after")
        return float(v) if v is not None else None
    except Exception:
        return None


def _try_model(model, messages, max_tokens, temperature, tools, tool_choice,
               reasoning_effort, max_attempts):
    client = get_client()  # 缺 GROQ_API_KEY 在這裡直接 KeyError,不要被當成暫時性錯誤重試
    current_max = max_tokens
    last = None
    for attempt in range(1, max_attempts + 1):
        kwargs = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=current_max,
        )
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        tag = f"[Groq] {model} {attempt}/{max_attempts}"

        try:
            resp = client.chat.completions.create(**kwargs)
        except APIStatusError as e:
            last = e
            status = e.status_code
            code = _error_code(e)
            print(f"{tag}: HTTP {status} {code} {str(e)[:200]!r}")

            if status == 400 and code == "tool_use_failed":
                # 模型沒照 tool_choice 呼叫工具 → 隨機失敗,重試
                time.sleep(2 + random.random() * 3)
                continue
            if status == 413:
                m = re.search(r"Limit (\d+), Requested (\d+)", str(e))
                new_max = (current_max - (int(m.group(2)) - int(m.group(1))) - 200
                           if m else current_max // 2)
                if new_max < _MIN_TOKENS:
                    raise GroqPermanentError(f"request too large even at minimum budget: {e}")
                print(f"{tag}: 413 → max_tokens {current_max} -> {new_max}")
                current_max = new_max
                continue
            if status == 429:
                ra = _retry_after(e)
                if ra is not None and ra > _QUOTA_WAIT_LIMIT:
                    raise GroqError(f"{model} quota exhausted (retry-after {ra:.0f}s)")
                wait = min((ra + 1) if ra is not None else 30 * attempt, 120)
                print(f"{tag}: 429 → wait {wait:.0f}s")
                time.sleep(wait + random.random())
                continue
            if status in (401, 403):
                raise GroqPermanentError(f"auth error {status}: {e}")
            if status == 404 or code in ("model_not_found", "model_decommissioned"):
                raise GroqError(f"{model} unavailable: {e}")
            if status in (400, 422):
                raise GroqPermanentError(f"bad request: {e}")
            time.sleep(min(2 ** attempt, 30))  # 408 / 409 / 5xx
            continue
        except (APIConnectionError, APITimeoutError) as e:
            last = e
            wait = min(2 ** attempt, 30)
            print(f"{tag}: network error {e!r}, retry in {wait}s")
            time.sleep(wait)
            continue

        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        finish = choice.finish_reason or "unknown"
        if text:
            return text, finish

        # 空輸出:最常見原因是 reasoning 把 max_completion_tokens 吃光
        last = RuntimeError(f"empty content (finish_reason={finish})")
        print(f"{tag}: empty content, finish_reason={finish}")
        if finish == "length":
            current_max = min(int(current_max * 1.5), _TOKEN_CEILING)
            print(f"{tag}: raise max_tokens -> {current_max}")
        time.sleep(2)

    raise GroqError(f"{model} failed after {max_attempts} attempts: {last}")


def _call_groq_raw(model, messages, max_tokens, temperature, tools=None,
                   tool_choice=None, reasoning_effort=None, max_attempts=5):
    """回傳 (text, finish_reason)。主 model 不行就換備援 model。"""
    models = [model]
    if FALLBACK_MODEL and FALLBACK_MODEL != model:
        models.append(FALLBACK_MODEL)
    last = None
    for i, m in enumerate(models):
        try:
            return _try_model(m, messages, max_tokens, temperature, tools, tool_choice,
                              reasoning_effort, max_attempts if i == 0 else 3)
        except GroqPermanentError:
            raise
        except GroqError as e:
            last = e
            if i + 1 < len(models):
                print(f"[Groq] {m} 失敗 -> 改用備援 {models[i + 1]}: {str(e)[:200]}")
    raise GroqError(f"all models failed: {last}")


def call_groq(model: str, system: str, user: str,
              max_tokens: int = 1500,
              temperature: float = 0.4,
              tools: list | None = None,
              tool_choice: str | None = None,
              reasoning_effort: str | None = None) -> str:
    """單次呼叫,只回傳 text。給短輸出 (critic / aggregator / picker) 用。
    保證回傳非空字串,否則 raise GroqError。"""
    text, finish = _call_groq_raw(
        model,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens, temperature, tools, tool_choice, reasoning_effort,
    )
    if finish == "length":
        print("[Groq] 注意:輸出撞到 max_tokens,內容可能被截斷")
    return text


_CONTINUE_PROMPT = (
    "你的輸出在中途被截斷了。請從最後一個字「無縫接著寫」剩下的部分,直到整篇結束。\n"
    "規則:1. 不要重複已寫過的內容。2. 不要任何開場白。"
    "3. 文章最後必須是一段「一句話總結:...」,然後停下。"
)


def _join(prev: str, chunk: str) -> str:
    """接續兩段文字;若模型把結尾又重複了一遍,去掉重疊部分。"""
    for n in range(min(200, len(prev), len(chunk)), 9, -1):
        if prev.endswith(chunk[:n]):
            return prev + chunk[n:]
    return prev + chunk


def call_groq_complete(model: str, system: str, user: str,
                       max_tokens: int = 4000,
                       temperature: float = 0.4,
                       max_continuations: int = 2,
                       tools: list | None = None,
                       tool_choice: str | None = None,
                       reasoning_effort: str | None = None) -> str:
    """長輸出用。只有 finish_reason=="length" (真的被截斷) 才續寫,最多 max_continuations 次。
    續寫用正式的多輪對話格式 (assistant 已寫的部分 + user 要求接續)。"""
    base = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    full, finish = _call_groq_raw(model, base, max_tokens, temperature, tools,
                                  tool_choice, reasoning_effort)
    print(f"[Groq] call 1: finish_reason={finish}, total={len(full)} chars")
    for i in range(max_continuations):
        if finish != "length":
            break
        print(f"[Groq] 被截斷 -> 續寫 ({i + 1}/{max_continuations})")
        messages = base + [
            {"role": "assistant", "content": full},
            {"role": "user", "content": _CONTINUE_PROMPT},
        ]
        chunk, finish = _call_groq_raw(model, messages, min(max_tokens, 2000),
                                       temperature, reasoning_effort=reasoning_effort)
        full = _join(full, chunk)
        print(f"[Groq] call {i + 2}: finish_reason={finish}, total={len(full)} chars")
    if finish == "length":
        print("[Groq] 續寫額度用盡,仍被截斷;交給後續驗證處理")
    return full


# ---------- 輸出標記判斷 (NO_ISSUES / ALL_CLEAN) ----------

def is_marker(text: str, marker: str) -> bool:
    """text 是否「只有」marker (容忍 markdown 符號、標點、大小寫、底線)。"""
    core = re.sub(r"[\s`*_>\"'「」.。!！]+", "", text or "").upper()
    return core == re.sub(r"[\s_]", "", marker).upper()


def truncate_review(text: str, max_chars: int = 1500) -> str:
    """Critic 若回 NO_ISSUES 就縮成單行,否則截斷。"""
    text = (text or "").strip()
    if is_marker(text, "NO_ISSUES"):
        return "NO_ISSUES"
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    if is_marker(first, "NO_ISSUES") and len(text) < 200:
        return "NO_ISSUES"
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n…(以下已截斷)"
