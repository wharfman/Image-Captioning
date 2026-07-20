'''
Environment-driven server configuration. Copy .env.example to .env and
override any of these; everything has a sane default for local dev so the
server runs with zero configuration out of the box.
'''
import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent  # backend/
REPO_ROOT = BACKEND_DIR.parent
load_dotenv(BACKEND_DIR / '.env')

CHECKPOINT_PATH = Path(os.getenv('CHECKPOINT_PATH', REPO_ROOT / 'checkpoints' / 'best_clip_caption.pt'))

# Comma-separated list of allowed frontend origins for CORS.
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000').split(',')
    if origin.strip()
]

MAX_UPLOAD_BYTES = int(os.getenv('MAX_UPLOAD_BYTES', 5 * 1024 * 1024))  # 5MB default
