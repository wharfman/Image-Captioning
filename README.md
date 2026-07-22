# Image Captioner

An end to end image captioning system. The application accepts an uploaded photograph and returns a short natural language description of its contents, for example "a brown dog running through the grass". The repository contains the model architecture, the full training pipeline, an HTTP API server for inference, and a web interface for uploading images and viewing results.

The generated captions are intended to be suitable for screen readers, making the project useful in accessibility contexts where images need to be described to users who cannot see them.

## Overview

The project is organized as two independent applications that communicate over HTTP:

* **backend/** (Python): the neural network, the training code, and a FastAPI server that accepts an image upload and responds with a caption.
* **frontend/** (TypeScript, React): a web interface where users upload an image, monitor progress, and receive an editable caption they can copy.

Both applications run locally. The frontend calls the backend API to perform inference.

## Model Architecture

Image captioning combines two tasks: visual understanding and text generation. This project assigns each task to a separate component.

### Visual encoder: frozen CLIP

Visual understanding is handled by CLIP (the ViT-L/14 variant), a vision model released by OpenAI and trained on a very large corpus of image and text pairs. CLIP is used as is. Its weights are frozen and are never updated during training.

The encoder extracts two outputs from CLIP rather than only its standard summary embedding:

1. A pooled feature vector. This is a single numerical representation of the entire image, used to initialize the hidden state of the text decoder.
2. A grid of patch features. The image is represented as a set of regions, each with its own feature vector. These allow the decoder to consult specific parts of the image while generating each word. The patch grid is average pooled by a factor of 2x2 (for example, 256 regions reduced to 64) to lower the cost of attention, which runs at every decoding step.

### Text decoder: GRU with attention

Text generation is handled by a GRU, a recurrent neural network that produces a caption one word at a time, with each word conditioned on the words generated before it. This is the component that is trained.

Two mechanisms improve output quality:

* **Attention.** Before predicting each word, the decoder computes a weighting over the image regions and forms a context vector from the most relevant ones. This allows the model to reference different parts of the image for different words, rather than relying on a single global summary throughout the sentence.
* **Beam search.** At inference time the model maintains the five highest scoring partial captions at each step and selects the best complete caption at the end, rather than committing to the single most likely word at every step (greedy decoding). Greedy decoding cannot revise an early mistake, while beam search can. Final scores are normalized by caption length, which prevents a bias toward short outputs.

### Vocabulary

A `Vocab` class converts between words and integer token IDs. It lowercases text, strips punctuation, and maintains special tokens for padding, start of sentence, end of sentence, and unknown words. The vocabulary built during training is stored inside the model checkpoint so the server always restores the exact mapping the model was trained with.

## Training Pipeline

The model is trained on Flickr30k, a public dataset of approximately 31,000 photographs, each annotated with about five human written captions. The dataset is not included in the repository and must be downloaded separately.

### Objective

Training uses teacher forcing: at each step the decoder receives the correct previous words from the reference caption and is trained to predict the next word. Prediction errors are measured with cross entropy loss (ignoring padding) and used to update the decoder weights. The encoder receives no gradient updates.

### Feature caching

Because the CLIP encoder is frozen, its output for a given image never changes. The training script therefore encodes every unique image exactly once and writes the results to a cache on disk. Subsequent epochs and subsequent runs read from the cache and never touch the raw images. This also avoids redundant work within a single epoch, since each image appears roughly five times (once per caption).

Patch features are stored in a NumPy file that is loaded memory mapped, so the operating system reads only the rows required by each batch instead of holding the entire multi gigabyte cache in RAM. Pooled features are small and remain in memory. An older single file cache format from an earlier version of the project is still recognized and loaded.

### Training loop features

* **Early stopping.** Training halts if validation loss fails to improve for 5 consecutive epochs.
* **Checkpointing.** Whenever validation loss reaches a new minimum, the model, optimizer state, epoch number, and vocabulary are saved to `checkpoints/best_clip_caption.pt`.
* **Mixed precision and gradient clipping.** Training runs under automatic mixed precision with a gradient scaler, and gradient norms are clipped for stability.
* **Logging.** Per epoch training and validation losses are appended to a CSV file, and a loss curve is plotted when training completes.
* **Sample output.** After each epoch the script prints a few generated captions alongside the corresponding reference captions for qualitative inspection.

### Hardware compatibility note

The decoder is implemented as a stack of `GRUCell` modules instead of the fused `nn.GRU` module. The fused kernel triggers a MIOpen error on some AMD ROCm configurations, while the cell based implementation uses standard matrix operations that run correctly on those systems. This makes the project trainable on AMD GPUs as well as NVIDIA GPUs.

## API Server

`backend/app/main.py` implements a FastAPI application. At startup it loads the checkpoint once, restores the vocabulary from it, and reconstructs the model using the architecture values defined in `captioning/config.py`. That config module is shared by the trainer and the server, so the two cannot diverge.

Endpoints:

* `GET /` returns a health check with the active device and vocabulary size.
* `POST /api/caption` accepts a multipart image upload, validates it (rejecting empty files, unreadable files, and uploads over the configured 5 MB limit), runs inference, and returns the caption as JSON.

Interactive API documentation is served at `http://localhost:8000/docs`. CORS is configured so the separately hosted frontend is permitted to call the API. The checkpoint path, allowed origins, and upload limit can all be overridden through a `.env` file read by `app/settings.py`.

## Web Frontend

The frontend is a Next.js 14 application written in TypeScript with React 18 and Tailwind CSS.

* **Upload.** A dropzone component supports drag and drop as well as click to browse, and validates files client side before any network request is made.
* **Request states.** The interface shows a loading placeholder while a caption is being generated, an error panel with a retry button when the request fails, and the result view on success.
* **Results.** The generated caption is displayed in an editable text field with copy to clipboard controls.
* **Resource management.** In flight requests are aborted when a new image is selected or the state is reset, and object URLs for image previews are revoked when replaced, preventing memory leaks across repeated uploads.

The API base URL defaults to `http://localhost:8000` and can be overridden with the `NEXT_PUBLIC_API_URL` environment variable.

## File Structure

```
Image-Captioning/
├── backend/
│   ├── requirements.txt          Python dependencies (PyTorch is installed separately, see notes in the file)
│   ├── train.py                  Training entrypoint: python train.py
│   ├── captioning/               Model library shared by training and serving
│   │   ├── config.py             Architecture settings (CLIP model, embedding size, hidden size, max length)
│   │   ├── model.py              EncoderCLIP, Attention, DecoderGRU with beam search, CaptioningModel
│   │   ├── vocab.py              Word to token ID conversion and decoding
│   │   ├── datasets.py           Caption file parsing, CLIP feature caching, PyTorch Dataset
│   │   └── training.py           Trainer class, dataloader construction, full training loop
│   └── app/                      API server
│       ├── main.py               FastAPI application, checkpoint loading, /api/caption endpoint
│       └── settings.py           Environment driven configuration (checkpoint path, CORS, upload limit)
├── frontend/
│   ├── app/
│   │   ├── page.tsx              Main page and upload/loading/result state handling
│   │   ├── layout.tsx            Root layout, font, and metadata
│   │   └── globals.css           Global styles
│   ├── components/
│   │   ├── Header.tsx            Title and description
│   │   ├── Dropzone.tsx          Drag and drop upload zone with validation
│   │   ├── ImagePreview.tsx      Uploaded image preview with file details and reset control
│   │   ├── CaptionSkeleton.tsx   Loading placeholder
│   │   └── CaptionResult.tsx     Editable caption output with copy controls
│   ├── lib/
│   │   ├── api.ts                API client with typed error handling
│   │   └── utils.ts              Client side file validation helpers
│   └── package.json, tsconfig.json, tailwind.config.ts, and related Next.js tooling
│
│   Created or expected at runtime, not committed:
├── flickr30k/                    Dataset: Images/ directory, captions.txt, and the CLIP feature cache
└── checkpoints/                  best_clip_caption.pt and loss_log.csv
```

## Setup and Usage

Requirements: Python 3.10 or later, Node.js 18 or later, and a GPU for training. Inference runs on CPU, though more slowly.

### 1. Backend installation

```bash
cd backend

# Install PyTorch first, selecting the correct build for your hardware
# (CUDA, ROCm, or CPU) at https://pytorch.org/get-started/locally/
# Then install the remaining dependencies:
pip install -r requirements.txt
```

### 2. Training

Download the Flickr30k dataset and arrange it in the repository root as `flickr30k/Images/` and `flickr30k/captions.txt`, then run:

```bash
python train.py
```

The first run encodes every image through CLIP and writes the feature cache, which takes some time. Later runs load the cache and begin training immediately. The best model is written to `checkpoints/best_clip_caption.pt`.

Training hyperparameters (batch size, learning rate, epoch count) are set at the top of `captioning/training.py`. Architecture settings live in `captioning/config.py`. Note that changing architecture settings invalidates existing checkpoints, which will no longer load until the model is retrained.

### 3. Running the API

```bash
cd backend
uvicorn app.main:app --port 8000
```

Optionally place a `.env` file in `backend/` to override the checkpoint path, allowed CORS origins, or the upload size limit.

### 4. Running the frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` in a browser and upload an image. If the API is hosted somewhere other than `localhost:8000`, set `NEXT_PUBLIC_API_URL` accordingly.

## Design Notes

* **Frozen encoder, trained decoder.** Reusing a pretrained CLIP encoder means only the comparatively small decoder needs training, which keeps the project feasible on a single consumer GPU.
* **Single source of configuration.** `captioning/config.py` defines the architecture for both training and serving, preventing a checkpoint from silently mismatching the server that loads it.
* **Cache first training.** Precomputing CLIP features reduces each epoch to inexpensive lookups, and memory mapped storage keeps RAM usage modest even with large caches.
* **Decoupled frontend and backend.** The interface and the model server are independent applications connected only by HTTP, so either can be modified, replaced, or deployed separately.
* **Flickr30k Dataset.** Excels at identifying realistic, everyday human activities but fails at recognizing specialized, abstract, or highly cluttered scenes due to image constraints.

## Technology Summary

| Layer | Technology |
|---|---|
| Visual encoder | OpenAI CLIP, ViT-L/14, frozen |
| Caption decoder | GRU with additive attention and beam search (PyTorch) |
| Training data | Flickr30k, approximately 31,000 images with 5 captions each |
| API server | FastAPI with Uvicorn |
| Frontend | Next.js 14, React 18, TypeScript, Tailwind CSS, lucide-react icons |
