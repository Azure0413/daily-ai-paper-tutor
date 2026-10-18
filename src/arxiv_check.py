"""用 arXiv API 驗證 paper 真的存在,並取得權威 metadata (標題 / 作者 / 摘要 / comment)。

LLM 的 browser_search 會編出不存在的 paper (標題、作者、會議混搭)。
這裡的結果 *不經過 LLM*,所以文章的出處行與去重 key 都以它為準。

回傳語意:
- 找到且標題吻合   -> ArxivMeta
- 確定沒有 / 對不上 -> None            (呼叫端應拒絕這個候選)
- 連不上 arXiv      -> raise ArxivUnavailable (呼叫端可降級,但要在文章標註「未驗證」)
"""
import os
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import requests

API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "ax": "http://arxiv.org/schemas/atom"}
TITLE_MATCH = 0.88
_MIN_INTERVAL = 3.0  # arXiv API 禮儀:請求間隔 >= 3 秒
_last_call = 0.0


class ArxivUnavailable(RuntimeError):
    pass


@dataclass
class ArxivMeta:
    arxiv_id: str
    title: str
    abstract: str
    authors: list = field(default_factory=list)
    comment: str = ""


def norm_title(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", s)


def title_matches(claimed: str, real: str) -> bool:
    a, b = norm_title(claimed), norm_title(real)
    if not a or not b:
        return False
    if SequenceMatcher(None, a, b).ratio() >= TITLE_MATCH:
        return True
    # 模型常省略副標題:較短者夠長且被較長者包含,也算
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 12 and short in long_


def clean_id(raw) -> str | None:
    if not raw:
        return None
    m = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?", str(raw))
    return m.group(1) if m else None


def _get(params: dict) -> ET.Element:
    global _last_call
    repo = os.environ.get("GITHUB_REPOSITORY", "local")
    headers = {"User-Agent": f"daily-ai-paper-tutor/1.0 (+https://github.com/{repo})"}
    last = "unknown"
    for attempt in range(3):
        wait = _MIN_INTERVAL - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(API, params=params, headers=headers, timeout=30)
            _last_call = time.time()
            if r.status_code == 200:
                return ET.fromstring(r.content)
            last = f"HTTP {r.status_code}"
        except (requests.RequestException, ET.ParseError) as e:
            _last_call = time.time()
            last = repr(e)
        print(f"[arXiv] attempt {attempt + 1}/3 failed: {last}")
        time.sleep(5 * (attempt + 1))
    raise ArxivUnavailable(last)


def _clean(s: str | None) -> str:
    return " ".join((s or "").split())


def parse_feed(root: ET.Element) -> list[ArxivMeta]:
    out = []
    for e in root.findall("a:entry", NS):
        id_url = _clean(e.findtext("a:id", "", NS))
        if "/api/errors" in id_url:  # 無效 id 時 arXiv 回傳的假 entry
            continue
        aid = re.sub(r"v\d+$", "", re.sub(r"^https?://arxiv\.org/abs/", "", id_url))
        out.append(ArxivMeta(
            arxiv_id=aid,
            title=_clean(e.findtext("a:title", "", NS)),
            abstract=_clean(e.findtext("a:summary", "", NS)),
            authors=[_clean(a.findtext("a:name", "", NS)) for a in e.findall("a:author", NS)],
            comment=_clean(e.findtext("ax:comment", "", NS)),
        ))
    return out


def lookup_paper(arxiv_id, title: str) -> ArxivMeta | None:
    aid = clean_id(arxiv_id)
    if aid:
        for m in parse_feed(_get({"id_list": aid, "max_results": 1})):
            if title_matches(title, m.title):
                return m
            print(f"[arXiv] {aid} 存在但標題對不上: {m.title!r} vs 模型宣稱 {title!r}")
    q = re.sub(r'["\\]', " ", title or "").strip()
    if q:
        for m in parse_feed(_get({"search_query": f'ti:"{q}"', "max_results": 5})):
            if title_matches(title, m.title):
                return m
    return None


def venue_in_comment(comment: str, venue: str) -> bool:
    """資訊用:arXiv comment 是否提到模型宣稱的會議 (作者自填,不一定有)。"""
    name = re.split(r"[\s'’]", (venue or "").strip())[0].lower()
    return bool(name) and name in (comment or "").lower()
