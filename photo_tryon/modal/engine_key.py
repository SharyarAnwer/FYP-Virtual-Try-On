"""Makes the engine's key once, stores it in Modal as the secret "photo-tryon-key", and prints the address
for the app. The key stays on this computer in ~/.photo_tryon_key and in Modal - never in the project.

usage: python engine_key.py            (prints the photo engine's address; makes the key the first time)
       python engine_key.py video      (prints the video engine's address: the same key, another app)
"""
import os
import secrets
import subprocess
import sys
from pathlib import Path

KEY_FILE = Path.home() / ".photo_tryon_key"

if KEY_FILE.exists():
    key = KEY_FILE.read_text().strip()
else:
    key = secrets.token_hex(12)
    KEY_FILE.write_text(key)
    done = subprocess.run([sys.executable, "-m", "modal", "secret", "create", "--force", "photo-tryon-key", f"TRYON_KEY={key}"],
                          capture_output=True, text=True)
    if done.returncode != 0:
        KEY_FILE.unlink()
        sys.exit("Could not store the key in Modal:\n" + done.stdout + done.stderr)

import modal  # noqa: E402
app_name = "photo-tryon-video-engine" if sys.argv[1:] == ["video"] else "photo-tryon-engine"
url = modal.Function.from_name(app_name, "web").get_web_url()
print(f"{url.rstrip('/')}/k/{key}")
