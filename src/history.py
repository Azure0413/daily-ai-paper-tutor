import json
import os
import re
import unicodedata
from datetime import date
from pathlib import Path

HISTORY_FILE = Path(os.environ.get("HISTORY_FILE", "topics_history.json"))
MAX_ENTRIES = 90


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", s)


def _topic_key(topic: str) -> str:
    """'Kernelized Attention (Spectraformer) – ICLR 2024' -> 'kernelizedattention'"""
    head = re.split(r"\s[–—\-]\s|[(（:：]", unicodedata.normalize("NFKC", topic or ""))[0]
    return _norm(head)


def _read() -> dict:
    if not HISTORY_FILE.exists():
        return {"topics": []}
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("topics"), list):
            raise ValueError("missing 'topics' list")
        return data
    except (ValueError, OSError) as e:
        # 不要默默當成空的再覆寫,否則歷史 (去重依據) 會被整個洗掉
        raise RuntimeError(f"{HISTORY_FILE} 內容毀損,請先修復再執行: {e}") from e


def load_entries() -> list[dict]:
    return [x for x in _read()["topics"] if isinstance(x, dict) and x.get("topic")]


def load_history() -> list[str]:
    return [x["topic"] for x in load_entries()]


def avoid_list(entries: list[dict], n: int = 40) -> list[str]:
    """給 prompt 用的精簡「已教過」清單:最近 n 筆、去重、每筆最多 50 字。"""
    seen, out = set(), []
    for x in reversed(entries):
        raw = unicodedata.normalize("NFKC", x.get("title") or re.split(r"\s[–—\-]\s", x["topic"])[0])
        label = re.sub(r"[*_`]+", "", raw).strip()[:50]
        k = _norm(label)
        if k and k not in seen:
            seen.add(k)
            out.append(label)
        if len(out) >= n:
            break
    return list(reversed(out))


def find_duplicate(entries: list[dict], arxiv_id: str | None, title: str) -> str | None:
    """回傳重複原因 (字串),不重複回 None。優先用 arXiv id,舊資料退回標題比對。"""
    nt = _norm(title)
    for x in entries:
        if arxiv_id and x.get("arxiv_id") == arxiv_id:
            return f"arXiv:{arxiv_id} 已在 {x.get('date')} 教過"
        for cand in (x.get("title"), _topic_key(x["topic"])):
            nc = _norm(cand) if cand else ""
            if len(nc) >= 8 and len(nt) >= 8 and (nc in nt or nt in nc):
                return f"與 {x.get('date')} 的「{x['topic'][:40]}」重複"
    return None


def save_topic(topic: str, arxiv_id: str | None = None, title: str | None = None) -> None:
    data = _read()
    entry = {"date": date.today().isoformat(), "topic": topic}
    if arxiv_id:
        entry["arxiv_id"] = arxiv_id
    if title:
        entry["title"] = title
    data["topics"].append(entry)
    data["topics"] = data["topics"][-MAX_ENTRIES:]
    tmp = HISTORY_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, HISTORY_FILE)  # 原子替換,寫到一半崩潰也不會留下壞檔


def extract_topic(article: str) -> str:
    """從文章第一個非空行抓「主題:xxx」,並清掉 markdown 符號與特殊空白 (U+202F 等)。"""
    for line in (article or "").splitlines():
        line = line.strip()
        if not line:
            continue
        line = unicodedata.normalize("NFKC", line)
        line = re.sub(r"^[\s*_#>`]+|[\s*_#>`]+$", "", line)
        m = re.match(r"^主題\s*[:：]\s*(.+)$", line)
        topic = re.sub(r"[*_`]+", "", m.group(1) if m else line).replace("\u2010", "-")
        topic = re.sub(r"\s+", " ", topic).strip()
        return topic[:120] or "(untitled)"
    return "(untitled)"
