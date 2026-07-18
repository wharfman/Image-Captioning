import os
import csv
import random
from collections import Counter

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import clip
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Model configuration -- single source of truth.
# server.py imports these, so changing them here is the ONLY place they need
# to change. Note: they describe the architecture a checkpoint was trained
# with; if you change them, existing checkpoints will no longer load until
# you retrain (the weight shapes must match).
# ---------------------------------------------------------------------------
CLIP_MODEL = 'ViT-L/14'
EMBED_SIZE = 512
HIDDEN_SIZE = 1024
MAX_LEN = 30  # max generated caption length at inference


class Vocab:
    '''
    text pre-processing pipeline
    convert words in sentences to nums for nn to read
    '''
    def __init__(self, min_freq=1, max_size=None):
        self.min_freq = min_freq
        self.max_size = max_size
        self.word2indx = {}
        self.indx2word = {}

        # special tokens
        self.PAD = '<PAD>' # accounts for empty spaces
        self.SOS = '<SOS>' # start of sentence
        self.EOS = '<EOS>' # end of sentence
        self.UNK = '<UNK>' # replace unknown words with 'UNK'

    def build_vocab(self, sentences):
        '''        
        tokenize all sentences and count num of times each word appears
        '''
        counter = Counter()
        for s in sentences:
            tokens = self.tokenize(s)
            counter.update(tokens)

        # filter by min_freq
        items = [(w, c) for w, c in counter.items() if c >= self.min_freq]
        items.sort(key=lambda x: (-x[1], x[0]))
        if self.max_size:
            items = items[:self.max_size]

        indx = 0
        for sp in [self.PAD, self.SOS, self.EOS, self.UNK]:
            self.word2indx[sp] = indx
            self.indx2word[indx] = sp
            indx +=1

        for w, _ in items:
            if w in self.word2indx:
                continue
            self.word2indx[w] = indx
            self.indx2word[indx] = w
            indx += 1

    def tokenize(self, s):
        # tokenizer: lowercase + split on spaces, strip punctuation
        s = s.lower().strip()

        # replace common punctuation with space
        for char in [".", ",", "!", "?", ";", ":", '"', "'", "(", ")"]:
            s = s.replace(char, ' ')

        tokens = [t for t in s.split() if t]
        return tokens
    
    def encode(self, s):
        # convert sentence into list of nums
        tokens = [self.SOS] + self.tokenize(s) + [self.EOS]
        ids = [self.word2indx.get(t, self.word2indx[self.UNK]) for t in tokens]
        return ids
    
    def decode(self, ids):
        words = []
        for i in ids:
            w = self.indx2word.get(i, self.UNK)
            if w == self.EOS:
                break # stop due to end of sentence found
            if w in (self.SOS, self.PAD):
                continue # skip special tokens
            words.append(w)
        return ' '.join(words)
    
    def __len__(self):
        return len(self.word2indx)
    
class FlickrDataset(Dataset):
    '''
    process Flickr-style (image,caption) dataset
    '''
    def __init__(self, images_root, captions_file, vocab, clip_preprocess, transform=None, max_caption_len=30, subset=None):
        self.images_root = images_root
        self.captions_file = captions_file
        self.vocab = vocab
        self.clip_preprocess = clip_preprocess
        self.transform = transform
        self.max_caption_len = max_caption_len

        self.items = []
        self._load_captions()
        if subset is not None and subset < len(self.items):
            random.seed(42)
            self.items = random.sample(self.items, subset)

    def _load_captions(self):
        with open(self.captions_file, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header row: "image,caption"
            for row in reader:
                if len(row) != 2:
                    continue
                img_name, caption = row
                img_name = img_name.strip()
                caption = caption.strip()
                if not img_name or not caption:
                    continue
                img_path = os.path.join(self.images_root, img_name)
                if not os.path.exists(img_path):
                    continue
                self.items.append((img_path, caption))

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, indx):
        img_path, caption = self.items[indx]
        image = Image.open(img_path).convert('RGB')
        image = self.clip_preprocess(image)

        # caption -> ids
        ids = self.vocab.encode(caption)
        if len(ids) > self.max_caption_len:
            ids = ids[:self.max_caption_len - 1] + [self.vocab.word2indx[self.vocab.EOS]]
        return image, torch.tensor(ids, dtype=torch.long) 
    

    @staticmethod
    def collate_fn(batch):
        images, captions = zip(*batch)
        images = torch.stack(images, dim=0)
        lengths = [len(c) for c in captions]
        max_len = max(lengths)
        padded = torch.full((len(captions), max_len), fill_value=0, dtype=torch.long)
        for i, c in enumerate(captions):
            padded[i, :len(c)] = c
        return images, padded, torch.tensor(lengths, dtype=torch.long)
    
class CachedFeatureDataset(Dataset):
    '''
    Serves precomputed CLIP features instead of raw images. The CLIP encoder
    is frozen, so its output for a given image never changes: computing it
    once and caching it removes the encoder (and all image loading/decoding)
    from the per-epoch cost entirely, with zero accuracy impact. It also
    avoids re-encoding the same image ~5x per epoch (Flickr30k has ~5
    captions per image).
    '''
    def __init__(self, items, name2indx, patch_feats, pooled_feats, vocab, max_caption_len=30):
        # items: list of (img_name, caption)
        # patch_feats: (num_unique_images, N, D) float16 CPU tensor
        # pooled_feats: (num_unique_images, D) float16 CPU tensor
        self.items = items
        self.name2indx = name2indx
        self.patch_feats = patch_feats
        self.pooled_feats = pooled_feats
        self.vocab = vocab
        self.max_caption_len = max_caption_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, indx):
        img_name, caption = self.items[indx]
        fi = self.name2indx[img_name]
        pf = self.patch_feats[fi]
        if isinstance(pf, torch.Tensor):          # legacy in-RAM cache
            patches = pf.to(torch.float32)
        else:                                     # numpy memmap row: copy just
            patches = torch.from_numpy(np.array(pf)).to(torch.float32)  # this row off disk
        pooled = self.pooled_feats[fi].to(torch.float32)

        ids = self.vocab.encode(caption)
        if len(ids) > self.max_caption_len:
            ids = ids[:self.max_caption_len - 1] + [self.vocab.word2indx[self.vocab.EOS]]
        return patches, pooled, torch.tensor(ids, dtype=torch.long)

    @staticmethod
    def collate_fn(batch):
        patches, pooled, captions = zip(*batch)
        patches = torch.stack(patches, dim=0)
        pooled = torch.stack(pooled, dim=0)
        lengths = [len(c) for c in captions]
        max_len = max(lengths)
        padded = torch.full((len(captions), max_len), fill_value=0, dtype=torch.long)
        for i, c in enumerate(captions):
            padded[i, :len(c)] = c
        return patches, pooled, padded, torch.tensor(lengths, dtype=torch.long)


def load_caption_items(images_root, captions_file):
    '''Parse captions.txt into a list of (img_name, caption), skipping missing images.'''
    items = []
    with open(captions_file, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header row: "image,caption"
        for row in reader:
            if len(row) != 2:
                continue
            img_name, caption = row
            img_name = img_name.strip()
            caption = caption.strip()
            if not img_name or not caption:
                continue
            if not os.path.exists(os.path.join(images_root, img_name)):
                continue
            items.append((img_name, caption))
    return items


@torch.no_grad()
def precompute_clip_features(encoder, images_root, image_names, cache_path, batch_size=64):
    '''
    Run every unique image through the frozen CLIP encoder exactly once and
    cache the features to disk. Subsequent runs load the cache instead of
    recomputing (delete the cache files if you change the CLIP model or the
    patch pooling).

    Patch features are stored in a separate .npy file written incrementally
    and loaded MEMORY-MAPPED, so the OS pages in only the rows each batch
    actually touches rather than holding the whole file resident in RAM.
    Pooled features are comparatively small and stay in RAM via the small
    .pt metadata file.

    Legacy caches (single .pt containing 'patch_feats', from an early
    Flickr8k-era format) are still recognized and loaded fully into RAM as
    before.
    '''
    patches_path = cache_path + '.patches.npy'

    if os.path.exists(cache_path):
        meta = torch.load(cache_path, map_location='cpu')
        if meta.get('image_names') == image_names:
            if 'patch_feats' in meta:  # legacy small cache: everything in one .pt
                print(f"Loading cached CLIP features (legacy format) from {cache_path}")
                return meta['patch_feats'], meta['pooled_feats'], meta['name2indx']
            print(f"Loading cached CLIP features (memory-mapped) from {cache_path}")
            patches = np.load(patches_path, mmap_mode='r')
            return patches, meta['pooled_feats'], meta['name2indx']
        print("Cache does not match current image list, recomputing...")

    preprocess = encoder.preprocess
    device = encoder.device
    n = len(image_names)
    if n == 0:
        raise RuntimeError(
            "precompute_clip_features received an empty image list -- "
            "no images to encode. Check the dataset paths/parsing upstream.")
    patches_mm = None
    pooled_feats = None

    for i in tqdm(range(0, n, batch_size), desc='Precomputing CLIP features (one-time)'):
        batch_names = image_names[i:i + batch_size]
        imgs = torch.stack([
            preprocess(Image.open(os.path.join(images_root, name)).convert('RGB'))
            for name in batch_names
        ], dim=0).to(device)
        patches, pooled = encoder(imgs)
        patches = patches.to(torch.float16).cpu().numpy()
        pooled = pooled.to(torch.float16).cpu()

        if patches_mm is None:
            # Shapes known after the first batch: preallocate the full file
            # on disk and fill it batch by batch.
            patches_mm = np.lib.format.open_memmap(
                patches_path, mode='w+', dtype=np.float16,
                shape=(n, patches.shape[1], patches.shape[2]),
            )
            pooled_feats = torch.empty(n, pooled.shape[1], dtype=torch.float16)

        patches_mm[i:i + len(batch_names)] = patches
        pooled_feats[i:i + len(batch_names)] = pooled

    patches_mm.flush()
    name2indx = {name: i for i, name in enumerate(image_names)}
    torch.save({
        'image_names': image_names,
        'pooled_feats': pooled_feats,
        'name2indx': name2indx,
        'patches_file': os.path.basename(patches_path),
    }, cache_path)
    size_gb = patches_mm.nbytes / 1e9
    print(f"Saved feature cache: {cache_path} + {patches_path} ({size_gb:.2f} GB patches, memory-mapped)")
    return np.load(patches_path, mmap_mode='r'), pooled_feats, name2indx


class EncoderCLIP(nn.Module):
    def __init__(self, device='cuda', clip_model='ViT-L/14'):
        super().__init__()
        self.device = device
        # ViT-L/14 gives richer features than ViT-B/32 (768-dim vs 512-dim,
        # 16x16 patch grid vs 7x7), at the cost of heavier (but frozen,
        # no-grad) encoder compute per batch.
        self.model, self.preprocess = clip.load(clip_model, device=device)
        self.feature_dim = self.model.visual.output_dim
        self.clip_name = clip_model.replace('/', '-')  # for cache filenames

        # freeze parameters
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, images):
        # images: preprocessed images tensor (B, 3, H, W)
        # Returns:
        #   patch_features:  (B, N, D) per-patch features, N = grid*grid,
        #                     used by the decoder's attention module so it
        #                     can look at different image regions per word.
        #   pooled_features: (B, D) the usual global CLIP embedding (same
        #                     value clip.encode_image would give), used only
        #                     to seed the decoder's initial hidden state.
        with torch.no_grad():
            visual = self.model.visual
            dtype = visual.conv1.weight.dtype
            x = images.type(dtype)

            x = visual.conv1(x)                          # (B, width, grid, grid)
            x = x.reshape(x.shape[0], x.shape[1], -1)     # (B, width, grid**2)
            x = x.permute(0, 2, 1)                        # (B, grid**2, width)
            cls = visual.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], device=x.device, dtype=x.dtype
            )
            x = torch.cat([cls, x], dim=1)                # (B, grid**2+1, width)
            x = x + visual.positional_embedding.to(x.dtype)
            x = visual.ln_pre(x)
            x = x.permute(1, 0, 2)                        # NLD -> LND
            x = visual.transformer(x)
            x = x.permute(1, 0, 2)                        # LND -> NLD

            pooled = visual.ln_post(x[:, 0, :])           # (B, width)
            patches = visual.ln_post(x[:, 1:, :])         # (B, grid**2, width)
            if visual.proj is not None:
                pooled = pooled @ visual.proj             # (B, D)
                patches = patches @ visual.proj           # (B, grid**2, D)

            pooled = pooled / pooled.norm(dim=-1, keepdim=True)
            pooled = pooled.to(torch.float32)
            patches = patches.to(torch.float32)

            # Pool the patch grid 2x2 -> cuts patch count ~4x (e.g. 256 -> 64
            # for ViT-L/14). Attention runs at every decoding timestep, not
            # once per image, so this directly cuts that per-step cost.
            grid = int(patches.shape[1] ** 0.5)
            patches = patches.reshape(patches.shape[0], grid, grid, -1).permute(0, 3, 1, 2)
            patches = F.avg_pool2d(patches, kernel_size=2)
            patches = patches.permute(0, 2, 3, 1)
            patches = patches.reshape(patches.shape[0], -1, patches.shape[-1])

        return patches, pooled


class Attention(nn.Module):
    '''
    Additive (Bahdanau-style) attention over CLIP patch features, conditioned
    on the decoder's current hidden state. Lets the decoder look at different
    image regions for different words, instead of only seeing one pooled
    global vector for the whole image.
    '''
    def __init__(self, feature_dim, hidden_dim, attn_dim=256):
        super().__init__()
        self.feature_proj = nn.Linear(feature_dim, attn_dim)
        self.hidden_proj = nn.Linear(hidden_dim, attn_dim)
        self.full_att = nn.Linear(attn_dim, 1)

    def forward(self, features, hidden):
        # features: (B, N, D) patch features ; hidden: (B, H) top-layer hidden state
        att1 = self.feature_proj(features)                       # (B, N, attn_dim)
        att2 = self.hidden_proj(hidden).unsqueeze(1)              # (B, 1, attn_dim)
        scores = self.full_att(torch.tanh(att1 + att2)).squeeze(-1) # (B, N)
        alpha = torch.softmax(scores, dim=-1)                     # (B, N)
        context = (features * alpha.unsqueeze(-1)).sum(dim=1)     # (B, D)
        return context, alpha

class DecoderGRU(nn.Module):
    def __init__(self, vocab_size, feature_dim=512, embed_size=300, hidden_size=512, num_layers=1, dropout=0.3, attn_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.img2hidden = nn.Linear(feature_dim, hidden_size)
        self.attention = Attention(feature_dim, hidden_size, attn_dim)

        # Stack of GRUCells replaces nn.GRU. The fused nn.GRU kernel hits
        # MIOpen's RNN descriptor path, which crashes with
        # miopenStatusUnknownError on some ROCm setups. GRUCell instead uses
        # plain matmul/elementwise ops, which MIOpen handles without issue.
        # The first layer's input is now [word embedding ; attended context],
        # so it takes embed_size + feature_dim inputs instead of embed_size.
        self.cells = nn.ModuleList([
            nn.GRUCell(embed_size + feature_dim if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        # Dropout between stacked layers only, matching nn.GRU's own convention
        # (no dropout after the final layer's output).
        self.dropout = nn.Dropout(dropout) if num_layers > 1 else nn.Identity()
        # Output layer also sees the attended context directly (concatenated
        # with the hidden state), not just the hidden state alone.
        self.fc = nn.Linear(hidden_size + feature_dim, vocab_size)

    def _init_hidden(self, pooled_features):
        # Same initial hidden state broadcast to every layer, matching the
        # old h0.repeat(num_layers, 1, 1) behavior.
        h0 = torch.tanh(self.img2hidden(pooled_features)) # (B, H)
        return [h0.clone() for _ in range(self.num_layers)]

    def _step(self, input_t, hidden_states, patch_features):
        # input_t: (B, E) word embedding ; hidden_states: list of (B, H), one per layer
        # patch_features: (B, N, D)
        # Attention is computed from the top layer's *previous* hidden state,
        # then the resulting context is fed into the first GRUCell alongside
        # the word embedding, and again into the output layer.
        context, alpha = self.attention(patch_features, hidden_states[-1])
        x = torch.cat([input_t, context], dim=-1)
        new_hidden = []
        for i, cell in enumerate(self.cells):
            h = cell(x, hidden_states[i])
            new_hidden.append(h)
            x = h
            if i < self.num_layers - 1:
                x = self.dropout(x)
        out = torch.cat([x, context], dim=-1) # x is the output of the last layer, (B, H)
        return out, new_hidden, alpha

    def forward(self, captions, patch_features, pooled_features, lengths=None, teacher_forcing=True):
        # captions: (B, T) token ids including SOS at pos 0
        # patch_features: (B, N, D) ; pooled_features: (B, D)
        B, T = captions.size()
        device = captions.device
        embeddings = self.embedding(captions) # (B, T, E)

        hidden_states = self._init_hidden(pooled_features)
        outputs = torch.zeros(B, T, self.vocab_size, device=device)

        if teacher_forcing:
            # step through the sequence manually (was a single fused GRU call)
            for t in range(T - 1):
                input_t = embeddings[:, t, :] # (B, E)
                out, hidden_states, _ = self._step(input_t, hidden_states, patch_features)
                logits = self.fc(out) # (B, V)
                outputs[:, t + 1, :] = logits
            return outputs
        else:
            # step-by-step generation
            input_t = embeddings[:, 0, :] # SOS embedding, (B, E)
            for t in range(1, T):
                out, hidden_states, _ = self._step(input_t, hidden_states, patch_features)
                logit = self.fc(out) # (B, V)
                outputs[:, t, :] = logit

                next_token = logit.argmax(dim=-1)
                input_t = self.embedding(next_token)
            return outputs
        
    def generate(self, patch_features, pooled_features, max_len=30, sos_indx=1, eos_indx=2, device='cuda', beam_size=5):
        '''
        Beam search decoding (replaces greedy argmax).

        Greedy commits to the single locally-likeliest word at every step
        and can never reconsider -- if "black" edges out "brown" by a
        sliver when describing a brown dog, "black" wins forever. Beam
        search keeps the `beam_size` best partial sentences alive at each
        step and finally returns the best *complete* sentence by
        length-normalized log-probability, which reduces exactly those
        small local errors. beam_size=1 is equivalent to greedy.

        Returns the same format as before: a list (one per image) of token
        id lists, so vocab.decode works unchanged on each.
        '''
        if patch_features.dim() == 2:
            patch_features = patch_features.unsqueeze(0)
        if pooled_features.dim() == 1:
            pooled_features = pooled_features.unsqueeze(0)
        B = pooled_features.size(0)

        results = []
        # Beams are per-image, so decode one image at a time, treating its
        # live beams as a mini-batch through _step.
        for b in range(B):
            patches_1 = patch_features[b:b+1]   # (1, N, D)
            pooled_1 = pooled_features[b:b+1]   # (1, D)

            # Each beam: (tokens list, cumulative log-prob, hidden_states)
            init_hidden = self._init_hidden(pooled_1)
            beams = [([sos_indx], 0.0, init_hidden)]
            finished = []  # completed beams: (tokens, cumulative log-prob)

            for _ in range(max_len - 1):
                if not beams:
                    break
                k = len(beams)
                # Batch all live beams through one _step call.
                last_tokens = torch.tensor([bm[0][-1] for bm in beams], device=device)
                input_t = self.embedding(last_tokens)  # (k, E)
                hidden_batch = [
                    torch.cat([bm[2][layer] for bm in beams], dim=0)  # (k, H)
                    for layer in range(self.num_layers)
                ]
                patches_k = patches_1.expand(k, -1, -1)  # (k, N, D)

                out, new_hidden, _ = self._step(input_t, hidden_batch, patches_k)
                log_probs = torch.log_softmax(self.fc(out), dim=-1)  # (k, V)

                # Every (beam, next-word) continuation, scored by total log-prob.
                scores = log_probs + torch.tensor(
                    [bm[1] for bm in beams], device=device
                ).unsqueeze(1)  # (k, V)
                flat = scores.view(-1)
                top_scores, top_idx = flat.topk(min(beam_size, flat.numel()))

                new_beams = []
                V = log_probs.size(-1)
                for score, idx in zip(top_scores.tolist(), top_idx.tolist()):
                    beam_i, word = divmod(idx, V)
                    tokens = beams[beam_i][0] + [word]
                    hidden = [layer[beam_i:beam_i+1] for layer in new_hidden]
                    if word == eos_indx:
                        finished.append((tokens, score))
                    else:
                        new_beams.append((tokens, score, hidden))
                beams = new_beams

                # Enough complete candidates to choose from -- stop early.
                if len(finished) >= beam_size:
                    break

            # Any still-live beams count as candidates too (hit max_len).
            finished.extend((bm[0], bm[1]) for bm in beams)

            # Length-normalized score: without dividing by length, beam
            # search systematically favors shorter sentences (every added
            # word makes the log-prob more negative).
            best_tokens, _ = max(finished, key=lambda f: f[1] / len(f[0]))
            results.append(best_tokens)
        return results
    
class CaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, patch_feats, pooled_feats, captions, teacher_forcing=True):
        # Training path: consumes precomputed CLIP features directly (the
        # frozen encoder is bypassed entirely during training).
        return self.decoder(captions, patch_feats, pooled_feats, teacher_forcing=teacher_forcing)

    def generate_from_features(self, patch_feats, pooled_feats, max_len=30, sos_indx=1, eos_indx=2, device='cuda', beam_size=5):
        return self.decoder.generate(patch_feats, pooled_feats, max_len=max_len, sos_indx=sos_indx, eos_indx=eos_indx, device=device, beam_size=beam_size)

    def forward_images(self, images, captions, teacher_forcing=True):
        # Convenience path for raw images (e.g. inference on new data).
        patch_feats, pooled_feats = self.encoder(images)
        return self.decoder(captions, patch_feats, pooled_feats, teacher_forcing=teacher_forcing)

    def generate(self, images, max_len=30, sos_indx=1, eos_indx=2, device='cuda', beam_size=5):
        patch_feats, pooled_feats = self.encoder(images)
        gen = self.decoder.generate(patch_feats, pooled_feats, max_len=max_len, sos_indx=sos_indx, eos_indx=eos_indx, device=device, beam_size=beam_size)
        return gen

class Trainer:
    def __init__(self, model, vocab, device ='cude', save_dir='./checkpoints'):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.word2indx[self.vocab.PAD])
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self, dataloader, optimizer, epoch, clip_grad=5.0):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(
            dataloader, desc=f"Train Epoch {epoch}",
            bar_format='{desc}: {percentage:3.0f}%|{bar}| batch {n_fmt}/{total_fmt} • elapsed: {elapsed} • remaining: {remaining} • {rate_fmt}{postfix}',
        )

        scaler = torch.amp.GradScaler(device=self.device if self.device in ('cuda', 'cpu') else 'cuda')

        for patch_feats, pooled_feats, captions, lengths in pbar:
            patch_feats = patch_feats.to(self.device, non_blocking=True)
            pooled_feats = pooled_feats.to(self.device, non_blocking=True)
            captions = captions.to(self.device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device if self.device in ('cuda', 'cpu') else 'cuda'):
                outputs = self.model(patch_feats, pooled_feats, captions, teacher_forcing=True)
                targets = captions
                B, T, V = outputs.size()
                outputs_flat = outputs[:, 1:, :].contiguous().view(-1, V)
                targets_flat = targets[:, 1:].contiguous().view(-1)
                loss = self.criterion(outputs_flat, targets_flat)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()

            loss_val = loss.item()  # single CPU-GPU sync point, reused below
            total_loss += loss_val
            pbar.set_postfix({'loss': f'{loss_val:.4f}'})

        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        for patch_feats, pooled_feats, captions, lengths in tqdm(
            dataloader, desc='Valid',
            bar_format='{desc}: {percentage:3.0f}%|{bar}| batch {n_fmt}/{total_fmt} • elapsed: {elapsed} • remaining: {remaining} • {rate_fmt}{postfix}',
        ):
            patch_feats = patch_feats.to(self.device, non_blocking=True)
            pooled_feats = pooled_feats.to(self.device, non_blocking=True)
            captions = captions.to(self.device, non_blocking=True)
            outputs = self.model(patch_feats, pooled_feats, captions, teacher_forcing=True)
            B, T, V = outputs.size()
            outputs_flat = outputs[:, 1:, :].contiguous().view(-1, V)
            targets_flat = captions[:, 1:].contiguous().view(-1)
            loss = self.criterion(outputs_flat, targets_flat)
            total_loss += loss.item()
        return total_loss / len(dataloader)
    
    def save_checkpoint(self, epoch, optimizer, name='checkpoint.pt'):
        path = os.path.join(self.save_dir, f'{name}')
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'vocab': self.vocab.word2indx,
        }
        torch.save(state, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state'])
        print(f"Loaded checkpoint from {path}")

def build_vocab_from_sentences(sents, min_freq=1, max_size=None):
    vocab = Vocab(min_freq=min_freq, max_size=max_size)
    vocab.build_vocab(sents)
    print(f"Vocab size: {len(vocab)}")
    return vocab

def build_vocab_from_captions(captions_file, min_freq=1, max_size=None):
    sents = []
    with open(captions_file, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header row: "image,caption"
        for row in reader:
            if len(row) != 2:
                continue
            _, caption = row
            caption = caption.strip()
            if not caption:
                continue
            sents.append(caption)
    return build_vocab_from_sentences(sents, min_freq=min_freq, max_size=max_size)

def _make_cached_loaders(train_items, val_items, images_root, vocab, encoder, batch_size, cache_path):
    '''Shared tail: precompute/load features for all images, build loaders.'''
    image_names = sorted({name for name, _ in train_items} | {name for name, _ in val_items})
    patch_feats, pooled_feats, name2indx = precompute_clip_features(
        encoder, images_root, image_names, cache_path
    )
    train_ds = CachedFeatureDataset(train_items, name2indx, patch_feats, pooled_feats, vocab)
    val_ds = CachedFeatureDataset(val_items, name2indx, patch_feats, pooled_feats, vocab)
    # num_workers=0 on purpose: __getitem__ is an in-RAM lookup (or a tiny
    # memmap row read), so worker processes would only add IPC overhead.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=CachedFeatureDataset.collate_fn,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=CachedFeatureDataset.collate_fn,
        num_workers=0, pin_memory=True,
    )
    return train_loader, val_loader

def make_dataloaders(images_root, captions_file, vocab, encoder, batch_size=8, subset=None, cache_path='clip_features.pt'):
    '''Flickr-style loaders: one captions.txt, random 90/10 split.'''
    items = load_caption_items(images_root, captions_file)
    if subset is not None and subset < len(items):
        random.seed(42)
        items = random.sample(items, subset)

    random.shuffle(items)
    split = int(0.9 * len(items))
    train_items, val_items = items[:split], items[split:]
    return _make_cached_loaders(train_items, val_items, images_root, vocab, encoder, batch_size, cache_path)

def main():
    # Config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device', device)

    # Hyperparameters (architecture values come from the module-level
    # constants at the top of this file -- shared with server.py)
    epochs = 100
    batch_size = 512
    embed_size = EMBED_SIZE
    hidden_size = HIDDEN_SIZE
    lr = 3e-3
    subset = None
    save_dir = './checkpoints'

    # Encoder first: dataloaders need it to precompute/load features, and
    # the cache filename is keyed to the CLIP model name.
    encoder = EncoderCLIP(device=device, clip_model=CLIP_MODEL)

    data_root = './flickr30k'
    images_root = os.path.join(data_root, 'Images')
    captions_file_path = os.path.join(data_root, 'captions.txt')

    if not os.path.exists(images_root):
        raise FileNotFoundError(f"Images directory not found: {images_root}")
    if not os.path.exists(captions_file_path):
        raise FileNotFoundError(f"Captions file not found: {captions_file_path}")

    # min_freq=1, max_size=None: every word gets a real vocab entry --
    # sensible on a tiny dataset where vocabulary is scarce.
    vocab = build_vocab_from_captions(captions_file_path, min_freq=1, max_size=None)

    feature_cache = os.path.join(data_root, f'clip_features_{encoder.clip_name}.pt')
    train_loader, val_loader = make_dataloaders(
        images_root, captions_file_path, vocab, encoder, batch_size=batch_size, subset=subset, cache_path=feature_cache)

    # Model
    decoder = DecoderGRU(len(vocab), feature_dim=encoder.feature_dim, embed_size=embed_size, hidden_size=hidden_size)
    model = CaptioningModel(encoder, decoder).to(device)

    # Trainer and Optimizer
    trainer = Trainer(model, vocab, device=device, save_dir=save_dir)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    # Logfile
    log_file = os.path.join(save_dir, 'loss_log.csv')
    os.makedirs(save_dir, exist_ok=True)

    # Write header if file doesn;t exist
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss'])

    best_val_loss = float('inf')
    patience = 10
    counter = 0

    # Training Loop
    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(train_loader, optimizer, epoch)
        val_loss = trainer.validate(val_loader)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Log CSV
        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss])

        # Early stopping + checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            trainer.save_checkpoint(epoch, optimizer, name='best_clip_caption.pt')
            print(f"Validation loss decreased, checkpoint saved.")
        else:
            counter += 1
            print(f"No improvement in val loss. Counter: {counter}/{patience}")
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        # Demo captions
        model.eval()
        with torch.no_grad():
            for patch_feats, pooled_feats, captions, lengths in val_loader:
                patch_feats = patch_feats[:4].to(device)
                pooled_feats = pooled_feats[:4].to(device)
                gen_ids = model.generate_from_features(patch_feats, pooled_feats, max_len=20,
                                        sos_indx=vocab.word2indx[vocab.SOS],
                                        eos_indx=vocab.word2indx[vocab.EOS],
                                        device=device)
                for i in range(min(4, len(gen_ids))):
                    gen = vocab.decode(gen_ids[i])
                    gt = vocab.decode(captions[i].tolist())
                    print('GT :', gt)
                    print('PRED:', gen)
                    print('---')
                break

    print('Training finished.')

    # Plot losses
    df = pd.read_csv('checkpoints/loss_log.csv')
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()