import os
import sys
from pathlib import Path

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("DISCORD_WEBHOOK_URL", "https://discord.test/webhook")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
