import sys
from groq_client import call_groq, MODEL_WITH_SEARCH, MODEL_REASONING
from prompts import (
    GENERATOR_SYSTEM, generator_user_prompt,
    MATH_CRITIC_SYSTEM, math_critic_user,
    CONCEPT_CRITIC_SYSTEM, concept_critic_user,
    FORMAT_CRITIC_SYSTEM, format_critic_user,
    FINAL_REFINER_SYSTEM, final_refiner_user,
)
from history import load_history, save_topic, extract_topic
from discord_sender import send_to_discord


def run():
    history = load_history()
    print(f"[History] {len(history)} past topics loaded")

    # ---------- Round 1: Generate ----------
    print("[Round 1] Generating draft with web search…")
    draft = call_groq(
        model=MODEL_WITH_SEARCH,
        system=GENERATOR_SYSTEM,
        user=generator_user_prompt(history),
        temperature=0.7,
    )
    print(f"[Round 1] draft length = {len(draft)}")

    # ---------- Round 2: Math Critic ----------
    print("[Round 2] Math critique…")
    math_review = call_groq(
        model=MODEL_REASONING,
        system=MATH_CRITIC_SYSTEM,
        user=math_critic_user(draft),
        temperature=0.2,
    )
    print(f"[Round 2] math_review = {math_review[:200]}…")

    # ---------- Round 3: Concept Critic ----------
    print("[Round 3] Concept critique…")
    concept_review = call_groq(
        model=MODEL_REASONING,
        system=CONCEPT_CRITIC_SYSTEM,
        user=concept_critic_user(draft),
        temperature=0.2,
    )
    print(f"[Round 3] concept_review = {concept_review[:200]}…")

    # ---------- Round 4: Format Critic ----------
    print("[Round 4] Format critique…")
    format_review = call_groq(
        model=MODEL_REASONING,
        system=FORMAT_CRITIC_SYSTEM,
        user=format_critic_user(draft),
        temperature=0.2,
    )
    print(f"[Round 4] format_review = {format_review[:200]}…")

    # ---------- Round 5: Final Refine ----------
    print("[Round 5] Final integration…")
    final = call_groq(
        model=MODEL_REASONING,
        system=FINAL_REFINER_SYSTEM,
        user=final_refiner_user(draft, math_review, concept_review, format_review),
        temperature=0.3,
    )
    print(f"[Round 5] final length = {len(final)}")

    # ---------- Send & Save ----------
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
        # 失敗也送一則訊息給自己,免得默默掛掉沒發現
        import traceback, requests, os
        msg = f"⚠️ daily-ai-paper-tutor 失敗:\n```\n{traceback.format_exc()[-1500:]}\n```"
        try:
            requests.post(os.environ["DISCORD_WEBHOOK_URL"],
                          json={"content": msg}, timeout=15)
        except Exception:
            pass
        sys.exit(1)
