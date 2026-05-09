import json
import re
from datetime import date
from pathlib import Path

HISTORY_FILE = Path("topics_history.json")

def load_history() -> list[str]:
    if not HISTORY_FILE.exists():
        return []
    data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    return [item["topic"] for item in data.get("topics", [])]

def save_topic(topic: str) -> None:
    data = {"topics": []}
    if HISTORY_FILE.exists():
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    data["topics"].append({
        "date": date.today().isoformat(),
        "topic": topic,
    })
    # 只保留最近 90 筆,避免 prompt 太長
    data["topics"] = data["topics"][-90:]
    HISTORY_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def extract_topic(article: str) -> str:
    """從文章第一行抓 '主題:xxx'。"""
    first_line = article.strip().splitlines()[0]
    m = re.search(r"主題[:：]\s*(.+)", first_line)
    return m.group(1).strip() if m else first_line[:80]
