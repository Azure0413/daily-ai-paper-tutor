import json
import os
import re
import sys
import traceback

from arxiv_check import ArxivUnavailable, lookup_paper, venue_in_comment
from discord_sender import send_to_discord
from groq_client import (
    MODEL_REASONING, MODEL_WITH_SEARCH, SEARCH_TOOL_CHOICE, SEARCH_TOOLS,
    call_groq, call_groq_complete, is_marker, truncate_review,
)
from history import avoid_list, extract_topic, find_duplicate, load_entries, save_topic
from prompts import (
    AGGREGATOR_SYSTEM, CONCEPT_CRITIC_SYSTEM, FINAL_REFINER_SYSTEM, FORMAT_CRITIC_SYSTEM,
    GENERATOR_SYSTEM, MATH_CRITIC_SYSTEM, PICK_SYSTEM,
    aggregator_user, concept_critic_user, final_refiner_user, format_critic_user,
    generator_user_prompt, math_critic_user, pick_user_prompt,
)

MAX_PAPER_ATTEMPTS = 4   # 挑 paper → 驗證 → 去重 → 寫作,最多嘗試幾個候選
MIN_ARTICLE_CHARS = 800


class PipelineError(RuntimeError):
    pass


# ---------------------------------------------------------------- helpers

def parse_pick(text: str) -> dict | None:
    """從 picker 輸出抓第一個 JSON 物件;抓不到或沒有 title 回 None。"""
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or not str(d.get("title") or "").strip():
        return None
    return {
        "title": str(d["title"]).strip(),
        "arxiv_id": d.get("arxiv_id") if d.get("arxiv_id") not in (None, "null", "") else None,
        "venue": str(d.get("venue") or "").strip(),
    }


def validate_article(text: str) -> list[str]:
    """回傳「會讓文章不能發」的問題清單;空 list = 通過。"""
    problems = []
    t = (text or "").strip()
    if len(t) < MIN_ARTICLE_CHARS:
        problems.append(f"太短 ({len(t)} 字)")
    if t.upper().startswith("SKIP"):
        problems.append("模型回 SKIP")
    if "一句話總結" not in t:
        problems.append("缺「一句話總結」(可能被截斷)")
    elif len(t.rsplit("一句話總結", 1)[-1].strip()) < 10:
        problems.append("「一句話總結」被截斷")
    return problems


def ensure_topic_line(article: str, fallback_title: str) -> str:
    first = next((ln for ln in article.strip().splitlines() if ln.strip()), "")
    if re.match(r"^[\s*#>`_]*主題\s*[:：]", first):
        return article.strip()
    return f"主題:{fallback_title}\n{article.strip()}"


def paper_context(meta) -> str:
    if not meta:
        return ""
    return (f"標題: {meta.title}\narXiv: {meta.arxiv_id}\n"
            f"作者: {', '.join(meta.authors[:8])}\n摘要: {meta.abstract}")


def build_footer(ref: dict, meta, math_ok: bool, concept_ok: bool) -> str:
    if meta:
        src = f"📄 出處:{meta.title} — arXiv:{meta.arxiv_id} (https://arxiv.org/abs/{meta.arxiv_id})"
    else:
        src = f"📄 出處:{ref['title']} (⚠️ 無法連線 arXiv,出處未經驗證)"
    note = ("ℹ️ AI 生成內容,公式請對照原文確認" if math_ok and concept_ok
            else "⚠️ 部分自動審查未完成,公式與出處請自行對照原文")
    return f"\n\n{src}\n{note}"


def run_critic(name: str, max_chars: int = 1500, **kw) -> str | None:
    """單一 critic 失敗不該讓整個流程掛掉:回 None 代表「這份審查缺席」。"""
    try:
        return truncate_review(call_groq(**kw), max_chars=max_chars)
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] critic 失敗,略過這份審查: {str(e)[:200]}")
        return None


# ---------------------------------------------------------------- Round 1

def get_draft(entries: list[dict]) -> tuple[str, dict, object]:
    """挑 paper → arXiv 驗證 → 去重 → 寫草稿。回傳 (draft, ref, meta)。"""
    avoid = avoid_list(entries)
    rejected: list[str] = []

    for attempt in range(1, MAX_PAPER_ATTEMPTS + 1):
        print(f"[Round 1a] 挑 paper (候選 {attempt}/{MAX_PAPER_ATTEMPTS}, browser_search)…")
        raw = call_groq(
            model=MODEL_WITH_SEARCH,
            system=PICK_SYSTEM,
            user=pick_user_prompt(avoid, rejected),
            max_tokens=2000,
            temperature=0.7,
            tools=SEARCH_TOOLS,
            tool_choice=SEARCH_TOOL_CHOICE,
            reasoning_effort="low",
        )
        ref = parse_pick(raw)
        if not ref:
            print(f"[R1a] 無法解析 picker 輸出: {raw[:120]!r}")
            rejected.append("(格式錯誤)")
            continue
        print(f"[R1a] 候選: {ref}")

        # ---- 驗證:這篇 paper 真的存在嗎?(不經過 LLM) ----
        meta = None
        try:
            meta = lookup_paper(ref["arxiv_id"], ref["title"])
            if meta is None:
                print(f"[R1a] ✗ arXiv 查無此 paper,拒絕: {ref['title']}")
                rejected.append(ref["title"])
                continue
            print(f"[R1a] ✓ arXiv:{meta.arxiv_id} {meta.title!r} "
                  f"(comment 提到 {ref['venue']!r}: {venue_in_comment(meta.comment, ref['venue'])})")
        except ArxivUnavailable as e:
            print(f"[R1a] ⚠️ arXiv 連不上 ({e}),降級為「未驗證」模式")

        # ---- 去重 (以 arXiv id 為主) ----
        dup = find_duplicate(entries, meta.arxiv_id if meta else None,
                             meta.title if meta else ref["title"])
        if dup:
            print(f"[R1a] ✗ 重複,拒絕: {dup}")
            rejected.append(meta.title if meta else ref["title"])
            continue

        # ---- Round 1b:寫草稿 (不帶工具) ----
        print("[Round 1b] 寫草稿…")
        draft = call_groq_complete(
            model=MODEL_REASONING,
            system=GENERATOR_SYSTEM,
            user=generator_user_prompt(
                meta.title if meta else ref["title"], ref["venue"],
                meta.arxiv_id if meta else None,
                meta.authors if meta else [], meta.abstract if meta else ""),
            max_tokens=4000,
            temperature=0.5,
            max_continuations=2,
            reasoning_effort="low",
        )
        problems = validate_article(draft)
        if problems:
            print(f"[R1b] ✗ 草稿不合格 {problems},換下一個候選")
            rejected.append(meta.title if meta else ref["title"])
            continue
        print(f"[R1b] draft = {len(draft)} chars")
        return ensure_topic_line(draft, meta.title if meta else ref["title"]), ref, meta

    raise PipelineError(f"連續 {MAX_PAPER_ATTEMPTS} 個候選都不合格: {rejected}")


# ---------------------------------------------------------------- main flow

def run():
    entries = load_entries()
    print(f"[History] {len(entries)} past topics loaded")

    draft, ref, meta = get_draft(entries)

    # ---------- Round 2~4: 三個 critic (失敗的那份視為缺席,不中斷) ----------
    print("[Round 2] Math critic…")
    math_r = run_critic("R2", model=MODEL_REASONING, system=MATH_CRITIC_SYSTEM,
                        user=math_critic_user(draft), max_tokens=2500,
                        temperature=0.2, reasoning_effort="medium")
    print("[Round 3] Concept critic…")
    concept_r = run_critic("R3", model=MODEL_REASONING, system=CONCEPT_CRITIC_SYSTEM,
                           user=concept_critic_user(draft, paper_context(meta)),
                           max_tokens=2000, temperature=0.2, reasoning_effort="low")
    print("[Round 4] Format critic…")
    format_r = run_critic("R4", max_chars=800, model=MODEL_REASONING,
                          system=FORMAT_CRITIC_SYSTEM, user=format_critic_user(draft),
                          max_tokens=1200, temperature=0.2, reasoning_effort="low")
    reviews = {"math": math_r, "concept": concept_r, "format": format_r}
    for k, v in reviews.items():
        print(f"[review:{k}] " + ("(缺席)" if v is None else f"{len(v)} chars"))

    issues = {k: v for k, v in reviews.items() if v is not None and not is_marker(v, "NO_ISSUES")}

    final = draft
    if not issues:
        print("[Early exit] 沒有任何 critic 提出問題,直接使用草稿")
    else:
        print("[Round 4.5] Aggregate critiques…")
        shown = lambda k: reviews[k] if reviews[k] is not None else "此審查未完成"  # noqa: E731
        action_list = call_groq(
            model=MODEL_REASONING, system=AGGREGATOR_SYSTEM,
            user=aggregator_user(draft, shown("math"), shown("concept"), shown("format")),
            max_tokens=1000, temperature=0.2, reasoning_effort="low",
        )
        print(f"[R4.5] action_list = {len(action_list)} chars")
        if is_marker(action_list, "ALL_CLEAN"):
            print("[Early exit @ aggregator] ALL_CLEAN")
        else:
            print("[Round 5] Final refinement…")
            refined = call_groq_complete(
                model=MODEL_REASONING, system=FINAL_REFINER_SYSTEM,
                user=final_refiner_user(draft, action_list),
                max_tokens=4000, temperature=0.3, max_continuations=2,
                reasoning_effort="low",
            )
            problems = validate_article(refined)
            if problems:
                # refiner 輸出壞掉 (被截斷 / 太短 / SKIP) → 退回已通過驗證的草稿
                print(f"[R5] ✗ refiner 輸出不合格 {problems},退回草稿")
            else:
                final = ensure_topic_line(refined, meta.title if meta else ref["title"])
                print(f"[R5] final = {len(final)} chars")

    # ---------- 最後一道閘門 ----------
    problems = validate_article(final)
    if problems:
        raise PipelineError(f"最終文章不合格,不發送: {problems}")

    final += build_footer(ref, meta, math_ok=math_r is not None, concept_ok=concept_r is not None)

    # ---------- Send & save (先送出成功,才寫歷史) ----------
    topic = extract_topic(final)
    print(f"[Send] topic = {topic}")
    send_to_discord(final)
    save_topic(topic, arxiv_id=meta.arxiv_id if meta else None,
               title=meta.title if meta else ref["title"])
    print("[Done]")


def _notify_failure() -> None:
    import requests
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        return
    msg = f"⚠️ daily-ai-paper-tutor 失敗:\n```\n{traceback.format_exc()[-1200:]}\n```"
    try:
        requests.post(url, json={"content": msg[:1990], "allowed_mentions": {"parse": []}},
                      timeout=15)
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    try:
        run()
    except Exception:  # noqa: BLE001
        _notify_failure()
        traceback.print_exc()
        sys.exit(1)
