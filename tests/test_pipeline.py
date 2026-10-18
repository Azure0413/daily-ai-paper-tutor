import json
import time
import types

import httpx
import pytest
from groq import APIStatusError, BadRequestError, RateLimitError

import arxiv_check
import discord_sender
import groq_client as gc
import history
import main as app


# ============================================================ fakes

def _resp(status, body=None, headers=None):
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return httpx.Response(status, request=req, json=body or {}, headers=headers or {})


def err(cls, status, code="", message="boom", headers=None):
    body = {"error": {"message": message, "type": "x", "code": code}}
    return cls(f"Error code: {status} - {body}", response=_resp(status, body, headers), body=body["error"])


def ok(text, finish="stop"):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(
        message=types.SimpleNamespace(content=text), finish_reason=finish)])


class FakeGroq:
    """script: list of (str | Exception | callable(kwargs)->response)。依序消耗。"""
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0)
        if callable(item):
            item = item(kw)
        if isinstance(item, Exception):
            raise item
        return item if not isinstance(item, str) else ok(item)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)


def install(monkeypatch, script):
    fake = FakeGroq(script)
    monkeypatch.setattr(gc, "get_client", lambda: fake)
    return fake


# ============================================================ groq_client

def test_tool_use_failed_is_retried_not_aborted(monkeypatch):
    """今天的實際失敗:400 tool_use_failed 不能被當成永久錯誤。"""
    fake = install(monkeypatch, [
        err(BadRequestError, 400, "tool_use_failed", "Tool choice is required, but model did not call a tool"),
        ok('{"title": "x"}'),
    ])
    text = gc.call_groq("openai/gpt-oss-120b", "s", "u", tools=gc.SEARCH_TOOLS, tool_choice="required")
    assert text == '{"title": "x"}'
    assert len(fake.calls) == 2
    assert fake.calls[0]["tool_choice"] == "required"


def test_other_400_is_permanent_no_retry(monkeypatch):
    fake = install(monkeypatch, [err(BadRequestError, 400, "invalid_request_error"), ok("never")])
    with pytest.raises(gc.GroqPermanentError):
        gc.call_groq("m", "s", "u")
    assert len(fake.calls) == 1


def test_empty_content_with_length_raises_budget(monkeypatch):
    fake = install(monkeypatch, [ok("", "length"), ok("", "length"), ok("done")])
    assert gc.call_groq("m", "s", "u", max_tokens=1000) == "done"
    budgets = [c["max_completion_tokens"] for c in fake.calls]
    assert budgets == [1000, 1500, 2250]


def test_413_shrinks_by_reported_excess(monkeypatch):
    e = err(APIStatusError, 413, "", "Request too large ... (TPM): Limit 8000, Requested 8900, please reduce")
    fake = install(monkeypatch, [e, ok("fine")])
    assert gc.call_groq("m", "s", "u", max_tokens=3000) == "fine"
    assert fake.calls[1]["max_completion_tokens"] == 3000 - 900 - 200


def test_413_input_itself_too_large_gives_up(monkeypatch):
    e = err(APIStatusError, 413, "", "Limit 8000, Requested 34087")
    install(monkeypatch, [e, e, e, e, e])
    with pytest.raises(gc.GroqPermanentError):
        gc.call_groq("m", "s", "u", max_tokens=1000)


def test_429_short_retry_after_waits_and_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    install(monkeypatch, [err(RateLimitError, 429, "rate_limit_exceeded", headers={"retry-after": "7"}), ok("yo")])
    assert gc.call_groq("m", "s", "u") == "yo"
    assert any(7 <= s <= 9 for s in sleeps)


def test_daily_quota_429_falls_back_to_other_model(monkeypatch):
    quota = err(RateLimitError, 429, "rate_limit_exceeded", headers={"retry-after": "9000"})
    fake = install(monkeypatch, [quota, ok("from fallback")])
    assert gc.call_groq("openai/gpt-oss-120b", "s", "u") == "from fallback"
    assert [c["model"] for c in fake.calls] == ["openai/gpt-oss-120b", gc.FALLBACK_MODEL]


def test_all_models_fail_raises(monkeypatch):
    quota = err(RateLimitError, 429, "x", headers={"retry-after": "9000"})
    install(monkeypatch, [quota, quota])
    with pytest.raises(gc.GroqError):
        gc.call_groq("openai/gpt-oss-120b", "s", "u")


def test_missing_api_key_fails_fast_not_retried(monkeypatch):
    monkeypatch.setattr(gc, "_client", None)
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(KeyError):
        gc.call_groq("m", "s", "u")


def test_continuation_only_when_truncated(monkeypatch):
    fake = install(monkeypatch, [ok("完整文章。一句話總結:好。", "stop")])
    out = gc.call_groq_complete("m", "s", "u")
    assert out == "完整文章。一句話總結:好。" and len(fake.calls) == 1


def test_continuation_when_length_and_overlap_trimmed(monkeypatch):
    first = "A" * 300 + "這是被截斷前的最後一段內容共二十字左右喔"
    second = "這是被截斷前的最後一段內容共二十字左右喔" + "接著寫完。一句話總結:完。"
    fake = install(monkeypatch, [ok(first, "length"), ok(second, "stop")])
    out = gc.call_groq_complete("m", "s", "u")
    assert out == first + "接著寫完。一句話總結:完。"
    msgs = fake.calls[1]["messages"]
    assert msgs[-2]["role"] == "assistant" and msgs[-1]["role"] == "user"


def test_continuation_capped(monkeypatch):
    fake = install(monkeypatch, [ok("a" * 50, "length")] * 3)
    gc.call_groq_complete("m", "s", "u", max_continuations=2)
    assert len(fake.calls) == 3


@pytest.mark.parametrize("t", ["NO_ISSUES", "NO_ISSUES.", "`NO_ISSUES`", "**NO_ISSUES**\n", "no_issues"])
def test_is_marker_true(t):
    assert gc.is_marker(t, "NO_ISSUES")


@pytest.mark.parametrize("t", ["", "問題 1: ...", "NO_ISSUES 但是第三步有錯", "ALL_CLEAN"])
def test_is_marker_false(t):
    assert not gc.is_marker(t, "NO_ISSUES")


def test_truncate_review():
    assert gc.truncate_review("NO_ISSUES\n(全部正確)") == "NO_ISSUES"
    assert gc.truncate_review("問題 1: " + "x" * 3000).endswith("(以下已截斷)")
    assert gc.truncate_review("") == ""


# ============================================================ history

def test_extract_topic_cleans_markdown_and_special_spaces():
    assert history.extract_topic("**主題: Expert Choice Routing\u202ffor MoE**\n內文") == "Expert Choice Routing for MoE"
    assert history.extract_topic("\n\n主題:δ‑GClip 收斂分析**\nx") == "δ-GClip 收斂分析"
    assert history.extract_topic("") == "(untitled)"
    assert history.extract_topic("沒有主題前綴的第一行") == "沒有主題前綴的第一行"


def test_find_duplicate_by_id_and_legacy_topic():
    entries = [
        {"date": "2026-10-01", "topic": "Kernelized Attention with Random Features (Spectraformer) – ICLR 2024"},
        {"date": "2026-10-02", "topic": "x", "arxiv_id": "2401.00001", "title": "Some Paper Title Here"},
    ]
    assert history.find_duplicate(entries, "2401.00001", "Totally Different")
    assert history.find_duplicate(entries, None, "Kernelized Attention with Random Features")
    assert history.find_duplicate(entries, None, "Flow Matching Done Right") is None


def test_avoid_list_is_compact_and_deduped():
    entries = [{"date": "d", "topic": f"Topic {i % 5} – detail {i}"} for i in range(90)]
    out = history.avoid_list(entries)
    assert len(out) == 5 and len(set(out)) == 5


def test_save_roundtrip_and_cap(tmp_path, monkeypatch):
    f = tmp_path / "h.json"
    monkeypatch.setattr(history, "HISTORY_FILE", f)
    for i in range(95):
        history.save_topic(f"t{i}", arxiv_id=f"2401.{i:05d}")
    es = history.load_entries()
    assert len(es) == 90 and es[-1]["arxiv_id"] == "2401.00094"
    assert not list(tmp_path.glob("*.tmp"))


def test_corrupt_history_raises_instead_of_wiping(tmp_path, monkeypatch):
    f = tmp_path / "h.json"
    f.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(history, "HISTORY_FILE", f)
    with pytest.raises(RuntimeError):
        history.save_topic("x")
    assert f.read_text(encoding="utf-8") == "{not json"


def test_real_world_history_sample_still_loads():
    from pathlib import Path
    # 用凍結的樣本 (非線上資料),避免歷史檔輪替後測試莫名失敗
    real = Path(__file__).resolve().parent / "fixtures" / "history_sample.json"
    entries = [x for x in json.loads(real.read_text(encoding="utf-8"))["topics"]]
    assert len(entries) == 90
    # 這份歷史裡 Kernelized Attention 出現 8 次 → 新機制必須擋得住
    assert history.find_duplicate(entries, None, "Kernelized Attention with Random Features")
    avoid = history.avoid_list(entries)
    assert len(avoid) <= 40
    old_chars = len("、".join(e["topic"] for e in entries))
    assert len("、".join(avoid)) < old_chars * 0.6  # prompt 變短,省 TPM


# ============================================================ arxiv_check

FEED = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
 <entry>
  <id>http://arxiv.org/abs/2401.12345v2</id>
  <title>Flow Matching for
   Generative Modeling: A Test</title>
  <summary>  We study things.
  And more. </summary>
  <author><name>Ada Lovelace</name></author><author><name>Alan Turing</name></author>
  <arxiv:comment>Accepted at ICLR 2025</arxiv:comment>
 </entry>
</feed>"""
ERR_FEED = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
 <id>http://arxiv.org/api/errors#incorrect_id_format_for_1</id><title>Error</title><summary>bad</summary></entry></feed>"""


def feed(xml):
    import xml.etree.ElementTree as ET
    return lambda params: ET.fromstring(xml)


def test_parse_feed_and_error_entry():
    import xml.etree.ElementTree as ET
    m, = arxiv_check.parse_feed(ET.fromstring(FEED))
    assert m.arxiv_id == "2401.12345"
    assert m.title == "Flow Matching for Generative Modeling: A Test"
    assert m.authors == ["Ada Lovelace", "Alan Turing"]
    assert m.abstract == "We study things. And more."
    assert "ICLR" in m.comment
    assert arxiv_check.parse_feed(ET.fromstring(ERR_FEED)) == []


def test_lookup_by_id_ok(monkeypatch):
    monkeypatch.setattr(arxiv_check, "_get", feed(FEED))
    m = arxiv_check.lookup_paper("arXiv:2401.12345v1", "Flow Matching for Generative Modeling: A Test")
    assert m and m.arxiv_id == "2401.12345"


def test_lookup_accepts_dropped_subtitle(monkeypatch):
    monkeypatch.setattr(arxiv_check, "_get", feed(FEED))
    assert arxiv_check.lookup_paper("2401.12345", "Flow Matching for Generative Modeling")


def test_lookup_rejects_fabricated_title(monkeypatch):
    monkeypatch.setattr(arxiv_check, "_get", feed(FEED))
    assert arxiv_check.lookup_paper("2401.12345", "Adaptive Step-Size Neural Ordinary Differential Equations") is None


def test_lookup_invalid_id_falls_back_to_title_search(monkeypatch):
    import xml.etree.ElementTree as ET
    def fake_get(params):
        return ET.fromstring(ERR_FEED if "id_list" in params else FEED)
    monkeypatch.setattr(arxiv_check, "_get", fake_get)
    m = arxiv_check.lookup_paper("9999.99999", "Flow Matching for Generative Modeling: A Test")
    assert m and m.arxiv_id == "2401.12345"


def test_arxiv_unreachable_raises(monkeypatch):
    import requests
    monkeypatch.setattr(arxiv_check.requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("x")))
    with pytest.raises(arxiv_check.ArxivUnavailable):
        arxiv_check.lookup_paper("2401.12345", "t")


# ============================================================ discord_sender

def test_sanitize_does_not_corrupt_inf_or_int():
    assert discord_sender.sanitize(r"x \in S, \inf_x f, \int f, \infty") == "x ∈ S, \\inf_x f, ∫ f, ∞"
    assert discord_sender.sanitize(r"$\frac{a}{b}$") == "(a)/(b)"


def test_split_chunks_limit_and_fence_balance():
    code = "```\n" + "line\n" * 600 + "```\n"
    chunks = discord_sender.split_chunks("前言\n" + code + "後記\n", limit=500)
    assert all(len(c) <= 520 for c in chunks)
    assert all(c.count("```") % 2 == 0 for c in chunks)


class FakePost:
    def __init__(self, statuses):
        self.statuses, self.payloads = list(statuses), []

    def __call__(self, url, json=None, timeout=None):
        self.payloads.append(json)
        code = self.statuses.pop(0) if self.statuses else 204
        return types.SimpleNamespace(status_code=code, text="", json=lambda: {"retry_after": 0.1})


def test_send_blocks_mentions_and_retries_429(monkeypatch):
    post = FakePost([429, 204])
    monkeypatch.setattr(discord_sender.requests, "post", post)
    discord_sender.send_to_discord("hi @everyone")
    assert post.payloads[0]["allowed_mentions"] == {"parse": []}
    assert len(post.payloads) == 2


def test_send_raises_when_discord_keeps_failing(monkeypatch):
    monkeypatch.setattr(discord_sender.requests, "post", FakePost([500] * 10))
    with pytest.raises(RuntimeError):
        discord_sender.send_to_discord("hello")


# ============================================================ main (end-to-end, mocked)

ARTICLE = ("主題:流匹配的速度場推導\n論文:Flow Matching for Generative Modeling (ICLR 2025)\n"
           + "這是一段很長的推導內容,用來湊足字數。" * 60 + "\n一句話總結:速度場回歸等價於條件流匹配。")
META = arxiv_check.ArxivMeta("2401.12345", "Flow Matching for Generative Modeling: A Test",
                             "We study things.", ["Ada", "Alan"], "Accepted at ICLR 2025")


class Env:
    """集中模擬 LLM / arXiv / Discord / 歷史檔。"""
    def __init__(self, monkeypatch, tmp_path, llm, lookup=None, entries=None):
        self.sent, self.llm_calls = [], []
        f = tmp_path / "h.json"
        f.write_text(json.dumps({"topics": entries or []}), encoding="utf-8")
        monkeypatch.setattr(history, "HISTORY_FILE", f)
        self.hist = f

        def fake_groq(model, system, user, **kw):
            self.llm_calls.append(system[:12])
            return llm(system, user)
        monkeypatch.setattr(app, "call_groq", fake_groq)
        monkeypatch.setattr(app, "call_groq_complete", fake_groq)
        monkeypatch.setattr(app, "lookup_paper", lookup or (lambda i, t: META))
        monkeypatch.setattr(app, "send_to_discord", lambda text: self.sent.append(text))

    def saved(self):
        return json.loads(self.hist.read_text(encoding="utf-8"))["topics"]


PICK = '{"title": "Flow Matching for Generative Modeling: A Test", "arxiv_id": "2401.12345", "venue": "ICLR 2025"}'


def clean_llm(system, user):
    if system.startswith("你是 AI 研究員。任務:用 browser_search"):
        return PICK
    if system.startswith("你是 AI 研究員,擅長"):
        return ARTICLE
    return "NO_ISSUES"


def test_happy_path_early_exit_and_history(monkeypatch, tmp_path):
    env = Env(monkeypatch, tmp_path, clean_llm)
    app.run()
    assert len(env.sent) == 1 and "arxiv.org/abs/2401.12345" in env.sent[0]
    assert "AI 生成內容" in env.sent[0]
    assert env.saved()[-1]["arxiv_id"] == "2401.12345"
    assert not any(s.startswith("你是 critique") for s in env.llm_calls)


def test_fabricated_paper_rejected_then_next_candidate(monkeypatch, tmp_path):
    picks = iter(['{"title": "Adaptive Step-Size Neural ODE", "arxiv_id": null, "venue": "NeurIPS 2024"}', PICK])
    def llm(system, user):
        return next(picks) if system.startswith("你是 AI 研究員。任務:用 browser") else clean_llm(system, user)
    def lookup(i, t):
        return None if "Adaptive" in t else META
    env = Env(monkeypatch, tmp_path, llm, lookup)
    app.run()
    assert env.saved()[-1]["arxiv_id"] == "2401.12345"


def test_duplicate_paper_rejected(monkeypatch, tmp_path):
    n = {"pick": 0}
    def llm(system, user):
        if system.startswith("你是 AI 研究員。任務:用 browser"):
            n["pick"] += 1
            return PICK
        return clean_llm(system, user)
    env = Env(monkeypatch, tmp_path, llm,
              entries=[{"date": "2026-10-01", "topic": "舊主題", "arxiv_id": "2401.12345"}])
    with pytest.raises(app.PipelineError):
        app.run()
    assert n["pick"] == app.MAX_PAPER_ATTEMPTS and env.sent == []
    assert len(env.saved()) == 1  # 失敗時不寫歷史


def test_arxiv_unreachable_degrades_with_warning(monkeypatch, tmp_path):
    def boom(i, t):
        raise arxiv_check.ArxivUnavailable("HTTP 503")
    env = Env(monkeypatch, tmp_path, clean_llm, boom)
    app.run()
    assert "出處未經驗證" in env.sent[0]


def test_refiner_garbage_falls_back_to_draft(monkeypatch, tmp_path):
    def llm(system, user):
        if system.startswith("你是 AI 研究員。任務:用 browser"):
            return PICK
        if system.startswith("你是 AI 研究員,擅長"):
            return ARTICLE
        if system.startswith("你是一位嚴格的數學教授"):
            return "問題 1: 第 3 步符號不一致"
        if system.startswith("你是 critique"):
            return "[數學][HIGH] 符號 → 統一"
        if system.startswith("你是最終編輯"):
            return "主題:截斷的版本\n寫到一半"
        return "NO_ISSUES"
    env = Env(monkeypatch, tmp_path, llm)
    app.run()
    assert "截斷的版本" not in env.sent[0] and "流匹配的速度場推導" in env.sent[0]


def test_aggregator_all_clean_variants_skip_refiner(monkeypatch, tmp_path):
    def llm(system, user):
        if system.startswith("你是 critique"):
            return "ALL_CLEAN."
        if system.startswith("你是最終編輯"):
            raise AssertionError("refiner should not run")
        if system.startswith("你是一位嚴格的數學教授"):
            return "問題 1: x"
        return clean_llm(system, user)
    env = Env(monkeypatch, tmp_path, llm)
    app.run()
    assert len(env.sent) == 1


def test_critic_failure_is_non_fatal_and_flagged(monkeypatch, tmp_path):
    def llm(system, user):
        if system.startswith("你是一位嚴格的數學教授"):
            raise gc.GroqError("math critic down")
        return clean_llm(system, user)
    env = Env(monkeypatch, tmp_path, llm)
    app.run()
    assert "部分自動審查未完成" in env.sent[0]


def test_truncated_draft_never_sent(monkeypatch, tmp_path):
    def llm(system, user):
        if system.startswith("你是 AI 研究員,擅長"):
            return "主題:x\n" + "很長的內容" * 300  # 沒有「一句話總結」= 被截斷
        return clean_llm(system, user)
    env = Env(monkeypatch, tmp_path, llm)
    with pytest.raises(app.PipelineError):
        app.run()
    assert env.sent == []


def test_skip_from_writer_moves_to_next_candidate(monkeypatch, tmp_path):
    arts = iter(["SKIP: 純 benchmark", ARTICLE])
    def llm(system, user):
        return next(arts) if system.startswith("你是 AI 研究員,擅長") else clean_llm(system, user)
    env = Env(monkeypatch, tmp_path, llm, lookup=lambda i, t: META)
    # 第二輪候選的 id 必須不同才不會被去重擋掉 → 用兩個不同 meta
    metas = iter([arxiv_check.ArxivMeta("2401.00001", "Benchmarky Paper", "a", [], ""), META])
    monkeypatch.setattr(app, "lookup_paper", lambda i, t: next(metas))
    app.run()
    assert env.saved()[-1]["arxiv_id"] == "2401.12345"


def test_log_scenario_end_to_end_through_real_client(monkeypatch, tmp_path):
    """重現使用者貼的 log:picker 第一次 400 tool_use_failed,之後整條 pipeline 要能跑完。"""
    script = [
        err(BadRequestError, 400, "tool_use_failed", "Tool choice is required, but model did not call a tool"),
        ok(PICK),                 # picker (retry)
        ok(ARTICLE),              # writer
        ok("NO_ISSUES"), ok("NO_ISSUES"), ok("NO_ISSUES"),  # 3 critics
    ]
    fake = install(monkeypatch, script)
    sent = []
    f = tmp_path / "h.json"
    f.write_text('{"topics": []}', encoding="utf-8")
    monkeypatch.setattr(history, "HISTORY_FILE", f)
    monkeypatch.setattr(app, "lookup_paper", lambda i, t: META)
    monkeypatch.setattr(app, "send_to_discord", sent.append)
    app.run()
    assert len(fake.calls) == 6 and len(sent) == 1
    assert json.loads(f.read_text(encoding="utf-8"))["topics"][-1]["arxiv_id"] == "2401.12345"
