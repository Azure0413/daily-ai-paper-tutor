import sys
from groq_client import (
    call_groq, call_groq_complete, truncate_review,
    MODEL_WITH_SEARCH, MODEL_REASONING,
)
from prompts import (
    GENERATOR_SYSTEM, generator_user_prompt,
    MATH_CRITIC_SYSTEM, math_critic_user,
    CONCEPT_CRITIC_SYSTEM, concept_critic_user,
    FORMAT_CRITIC_SYSTEM, format_critic_user,
    AGGREGATOR_SYSTEM, aggregator_user,
    FINAL_REFINER_SYSTEM, final_refiner_user,
)
from history import load_history, save_topic, extract_topic
from discord_sender import send_to_discord


def all_clean(reviews: list[str]) -> bool:
    """三份 critique 都沒問題就跳過 aggregation + refine (SID early exit)。"""
    return all(r.strip() == "NO_ISSUES" for r in reviews)


def run():
    history = load_history()
    print(f"[History] {len(history)} past topics loaded")

    # ---------- Round 1: Generator (web search,長輸出 -> 自動續寫) ----------
    print("[Round 1] Generate draft with web search…")
    draft = call_groq_complete(
        model=MODEL_WITH_SEARCH,
        system=GENERATOR_SYSTEM,
        user=generator_user_prompt(history),
        max_tokens=1800,
        temperature=0.7,
        max_continuations=3,
    )
    print(f"[R1] draft = {len(draft)} chars")

    # ---------- Round 2: Math Critic ----------
    print("[Round 2] Math critic…")
    math_r = truncate_review(call_groq(
        model=MODEL_REASONING,
        system=MATH_CRITIC_SYSTEM,
        user=math_critic_user(draft),
        max_tokens=900,
        temperature=0.2,
    ))
    print(f"[R2] math_review = {len(math_r)} chars")

    # ---------- Round 3: Concept Critic ----------
    print("[Round 3] Concept critic…")
    concept_r = truncate_review(call_groq(
        model=MODEL_REASONING,
        system=CONCEPT_CRITIC_SYSTEM,
        user=concept_critic_user(draft),
        max_tokens=900,
        temperature=0.2,
    ))
    print(f"[R3] concept_review = {len(concept_r)} chars")

    # ---------- Round 4: Format Critic ----------
    print("[Round 4] Format critic…")
    format_r = truncate_review(
        call_groq(
            model=MODEL_REASONING,
            system=FORMAT_CRITIC_SYSTEM,
            user=format_critic_user(draft),
            max_tokens=600,
            temperature=0.2,
        ),
        max_chars=800,
    )
    print(f"[R4] format_review = {len(format_r)} chars")

    # ---------- Early exit: 全部 NO_ISSUES 就跳過後兩輪 ----------
    if all_clean([math_r, concept_r, format_r]):
        print("[Early exit] All critics NO_ISSUES, skipping aggregation + refine")
        final = draft
    else:
        # ---------- Round 4.5: Aggregator ----------
        print("[Round 4.5] Aggregate critiques into action list…")
        action_list = call_groq(
            model=MODEL_REASONING,
            system=AGGREGATOR_SYSTEM,
            user=aggregator_user(draft, math_r, concept_r, format_r),
            max_tokens=600,
            temperature=0.2,
        )
        print(f"[R4.5] action_list = {len(action_list)} chars")

        if action_list.strip() == "ALL_CLEAN":
            print("[Early exit @ aggregator] ALL_CLEAN")
            final = draft
        else:
            # ---------- Round 5: Refiner (長輸出 -> 自動續寫) ----------
            print("[Round 5] Final refinement…")
            final = call_groq_complete(
                model=MODEL_REASONING,
                system=FINAL_REFINER_SYSTEM,
                user=final_refiner_user(draft, action_list),
                max_tokens=1800,
                temperature=0.3,
                max_continuations=3,
            )
            print(f"[R5] final = {len(final)} chars")

    # ---------- Send & save ----------
    topic = extract_topic(final)
    print(f"[Save] topic = {topic}")
    save_topic(topic)

    print("[Send] pushing to Discord…")
    send_to_discord(final)
    print("[Done]")


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        import traceback, requests, os
        msg = f"⚠️ daily-ai-paper-tutor 失敗:\n```\n{traceback.format_exc()[-1500:]}\n```"
        try:
            requests.post(
                os.environ["DISCORD_WEBHOOK_URL"],
                json={"content": msg},
                timeout=15,
            )
        except Exception:
            pass
        sys.exit(1)
