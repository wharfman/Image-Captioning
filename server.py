'''
Minimal fullstack captioning server.

Sits in the same folder as train_model.py, next to a static/index.html.

    pip install fastapi uvicorn python-multipart
    uvicorn server:app --port 8000

Then open http://localhost:8000 -- FastAPI serves the frontend AND the API.
'''
import io

import torch
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

# Reuse the exact classes AND configuration the checkpoint was trained
# with -- train_model.py is the single source of truth. Change EMBED_SIZE,
# HIDDEN_SIZE, CLIP_MODEL, or MAX_LEN there and this server picks them up
# automatically.
from train_model import (
    Vocab, EncoderCLIP, DecoderGRU, CaptioningModel,
    CLIP_MODEL, EMBED_SIZE, HIDDEN_SIZE, MAX_LEN,
)

CHECKPOINT = './checkpoints/best_clip_caption.pt'

# ---- Load everything ONCE at startup (same pattern as the Gradio app) ----
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Loading model on {device}...')

ckpt = torch.load(CHECKPOINT, map_location=device)

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

app = FastAPI(title='Image Captioner')


@app.get('/')
def home():
    # The entire frontend is one hand-written HTML file.
    return FileResponse('index.html')


@app.post('/api/caption')
def caption(image: UploadFile = File(...)):
    raw = image.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail='The uploaded file is empty.')
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