import os
import sys
from pathlib import Path

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("DISCORD_WEBHOOK_URL", "https://discord.test/webhook")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_real_arxiv_pool(monkeypatch):
    """預設讓 arXiv 候選池是空的 (測試不要真的連網);要測主路徑的測試自己再覆寫。"""
    import main as app
    monkeypatch.setattr(app, "fetch_candidates", lambda *a, **k: [])
