'''
Captioning API server.

The frontend is a separate Next.js app (see ../frontend) that talks to this
server over HTTP/CORS instead of being served from here.

    pip install -r requirements.txt
    uvicorn app.main:app --port 8000    # run from inside backend/

API docs (Swagger UI) live at http://localhost:8000/docs.
'''
import io
import sys
from pathlib import Path

# Allow `import captioning` to resolve regardless of how uvicorn was
# invoked (bare console script vs `python -m uvicorn`, different cwd, etc.)
# -- backend/ is captioning's parent, so put it on sys.path explicitly
# rather than relying on incidental cwd behavior.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Reuse the exact classes AND configuration the checkpoint was trained
# with -- captioning/ is the single source of truth. Change EMBED_SIZE,
# HIDDEN_SIZE, CLIP_MODEL, or MAX_LEN there and this server picks them up
# automatically.
from captioning.config import CLIP_MODEL, EMBED_SIZE, HIDDEN_SIZE, MAX_LEN
from captioning.vocab import Vocab
from captioning.model import EncoderCLIP, DecoderGRU, CaptioningModel

from app.settings import CHECKPOINT_PATH, CORS_ORIGINS, MAX_UPLOAD_BYTES

# ---- Load everything ONCE at startup (same pattern as the Gradio app) ----
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Loading model on {device}...')

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}\n"
        "Train a model first (python train.py) or set CHECKPOINT_PATH in .env "
        "to point at an existing checkpoint."
    )
ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

# Restore the vocab from the checkpoint -- never rebuild it from captions.txt.
vocab = Vocab()
vocab.word2indx = ckpt['vocab']
vocab.indx2word = {i: w for w, i in vocab.word2indx.items()}

encoder = EncoderCLIP(device=device, clip_model=CLIP_MODEL)
decoder = DecoderGRU(len(vocab), feature_dim=encoder.feature_dim,
                     embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE)
model = CaptioningModel(encoder, decoder).to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f'Model ready: vocab={len(vocab)}, device={device}')

app = FastAPI(title='Image Captioner API')

# The frontend runs on its own origin (Next.js dev server on :3000 by
# default, configurable via CORS_ORIGINS in .env) -- CORS must be explicit
# since this API no longer serves the frontend itself.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/')
def health():
    return {'status': 'ok', 'device': device, 'vocab_size': len(vocab)}


@app.post('/api/caption')
def caption(image: UploadFile = File(...)):
    raw = image.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail='The uploaded file is empty.')
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail='Image exceeds the 5MB upload limit.')
    try:
        pil = Image.open(io.BytesIO(raw)).convert('RGB')
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail='That file could not be read as an image.')

    # Same inference pipeline as the Gradio demo:
    # preprocess -> generate under no_grad -> decode ids to words.
    tensor = encoder.preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        gen_ids = model.generate(
            tensor,
            max_len=MAX_LEN,
            sos_indx=vocab.word2indx[vocab.SOS],
            eos_indx=vocab.word2indx[vocab.EOS],
            device=device,
        )
    return {'caption': vocab.decode(gen_ids[0])}
