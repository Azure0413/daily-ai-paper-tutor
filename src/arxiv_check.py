"""用 arXiv API 驗證 paper 真的存在,並取得權威 metadata (標題 / 作者 / 摘要 / comment)。

LLM 的 browser_search 會編出不存在的 paper (標題、作者、會議混搭)。
這裡的結果 *不經過 LLM*,所以文章的出處行與去重 key 都以它為準。

回傳語意:
- 找到且標題吻合   -> ArxivMeta
- 確定沒有 / 對不上 -> None            (呼叫端應拒絕這個候選)
- 連不上 arXiv      -> raise ArxivUnavailable (呼叫端可降級,但要在文章標註「未驗證」)
"""
import os
import random
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date
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


# ---------------------------------------------------------------- 候選池 (不經過 LLM)
# 為什麼要有這段:LLM 憑記憶 / browser_search 挑 paper,會一直繞回同一批「最有名」的論文
# (已教過 → 重複) 或編出不存在的編號 (2405.12345)。歷史越長,失敗率越高。
# 改成直接問 arXiv:「comment 欄位寫著已被某會議收錄」的 paper,產出的候選一定真實存在。
VENUE_NAMES = ("NeurIPS", "ICLR", "ICML", "AAAI")
FIRST_YEAR = 2024
TOPIC_KEYWORDS = (
    "score matching", "flow matching", "rectified flow", "consistency model", "diffusion sampler",
    "masked diffusion", "discrete diffusion", "normalizing flow", "variational inference",
    "Langevin", "optimal transport", "Schrodinger bridge", "stochastic differential equation",
    "neural ODE", "mixture of experts", "low-rank adaptation", "linear attention",
    "state space model", "policy gradient", "preference optimization", "gradient clipping",
    "Adam convergence", "optimizer", "random features", "kernel", "Bayesian inference",
    "generalization bound", "in-context learning", "contrastive learning", "scaling law",
    "quantization", "sampling", "Markov chain Monte Carlo", "implicit bias", "loss landscape",
)
_VENUE_RE = re.compile(r"\b(NeurIPS|NIPS|ICLR|ICML|CVPR|AAAI)\b[\s'’_\-:]*(?:20)?(2[4-9])\b", re.I)
_BAD_COMMENT_RE = re.compile(r"under review|submitted|workshop|rejected|withdrawn", re.I)
_MATH_HINTS = ("theor", "prove", "provabl", "deriv", "bound", "converge", "closed-form",
               "objective", "estimator", "variational", "unbiased", "likelihood", "optimal",
               "equivalen", "generaliz", "regret", "kernel", "differential equation",
               "sampler", "score matching", "minimax", "analysis")
_ENGINEERING_HINTS = ("benchmark", "dataset", "we release", "survey", "toolkit", "open-source")


def venue_from_comment(comment: str, today: date | None = None) -> str | None:
    """'Accepted at ICLR 2025' -> 'ICLR 2025';under review / workshop / 未來年份 -> None。"""
    c = comment or ""
    if _BAD_COMMENT_RE.search(c):
        return None
    m = _VENUE_RE.search(c)
    if not m:
        return None
    year = 2000 + int(m.group(2))
    if year > (today or date.today()).year:
        return None
    name = {"NIPS": "NeurIPS", "NEURIPS": "NeurIPS"}.get(m.group(1).upper(), m.group(1).upper())
    return f"{name} {year}"


def math_score(abstract: str) -> int:
    """很粗的啟發式:摘要含愈多「推導 / 理論」字眼愈高,純 benchmark / 資料集類扣分。"""
    a = (abstract or "").lower()
    return sum(w in a for w in _MATH_HINTS) - 2 * sum(w in a for w in _ENGINEERING_HINTS)


def fetch_candidates(n_queries: int = 4, per_query: int = 25, rng=None,
                     today: date | None = None) -> list[ArxivMeta]:
    """隨機「主題關鍵字 × 會議年份」查 arXiv,回傳 comment 證實已被收錄的 paper (去掉重複 id)。
    全部查詢都連不上才 raise ArxivUnavailable;部分成功就回傳已拿到的。"""
    rng = rng or random
    year_now = (today or date.today()).year
    out: dict[str, ArxivMeta] = {}
    last_err = None
    for _ in range(n_queries):
        kw = rng.choice(TOPIC_KEYWORDS)
        venue_q = f"{rng.choice(VENUE_NAMES)} {rng.randint(FIRST_YEAR, year_now)}"
        params = {
            "search_query": f'abs:"{kw}" AND co:"{venue_q}"',
            "start": rng.choice((0, 25, 50)),
            "max_results": per_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        try:
            feed = parse_feed(_get(params))
        except ArxivUnavailable as e:
            last_err = e
            continue
        for m in feed:
            if m.arxiv_id and venue_from_comment(m.comment, today) and m.arxiv_id not in out:
                out[m.arxiv_id] = m
        if len(out) >= 40:
            break
    if not out and last_err is not None:
        raise last_err
    return list(out.values())
