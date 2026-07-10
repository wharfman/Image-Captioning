import os   
import csv
import random
from collections import Counter

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
    
class Flickr8kDataset(Dataset):
    '''
    process flickr8k dataset
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
    avoids re-encoding the same image ~5x per epoch (Flickr8k has ~5
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
        patches = self.patch_feats[fi].to(torch.float32)
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
    cache the (patch, pooled) features to disk. Subsequent runs load the
    cache instead of recomputing (delete the cache file if you change the
    CLIP model or the patch pooling).
    '''
    if os.path.exists(cache_path):
        print(f"Loading cached CLIP features from {cache_path}")
        cache = torch.load(cache_path, map_location='cpu')
        if cache.get('image_names') == image_names:
            return cache['patch_feats'], cache['pooled_feats'], cache['name2indx']
        print("Cache does not match current image list, recomputing...")

    preprocess = encoder.preprocess
    device = encoder.device
    patch_list, pooled_list = [], []
    for i in tqdm(range(0, len(image_names), batch_size), desc='Precomputing CLIP features (one-time)'):
        batch_names = image_names[i:i + batch_size]
        imgs = torch.stack([
            preprocess(Image.open(os.path.join(images_root, n)).convert('RGB'))
            for n in batch_names
        ], dim=0).to(device)
        patches, pooled = encoder(imgs)
        # float16 halves cache size; converted back to float32 per-sample in
        # CachedFeatureDataset.__getitem__.
        patch_list.append(patches.to(torch.float16).cpu())
        pooled_list.append(pooled.to(torch.float16).cpu())

    patch_feats = torch.cat(patch_list, dim=0)
    pooled_feats = torch.cat(pooled_list, dim=0)
    name2indx = {n: i for i, n in enumerate(image_names)}
    torch.save({
        'image_names': image_names,
        'patch_feats': patch_feats,
        'pooled_feats': pooled_feats,
        'name2indx': name2indx,
    }, cache_path)
    size_gb = (patch_feats.numel() + pooled_feats.numel()) * 2 / 1e9
    print(f"Saved feature cache: {cache_path} ({size_gb:.2f} GB)")
    return patch_feats, pooled_feats, name2indx


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
        
    def generate(self, patch_features, pooled_features, max_len=30, sos_indx=1, eos_indx=2, device='cuda'):
        # greedy decoding for a single image/batch
        if patch_features.dim() == 2:
            patch_features = patch_features.unsqueeze(0)
        if pooled_features.dim() == 1:
            pooled_features = pooled_features.unsqueeze(0)
        B = pooled_features.size(0)
        generated = torch.full((B, max_len), fill_value=0, dtype=torch.long, device=device)
        hidden_states = self._init_hidden(pooled_features)
        input_t = self.embedding(torch.tensor([sos_indx]*B, device=device)) # (B, E)

        for t in range(1, max_len):
            out, hidden_states, _ = self._step(input_t, hidden_states, patch_features)
            logit = self.fc(out)
            next_token = logit.argmax(dim=-1)
            generated[:, t] = next_token
            input_t = self.embedding(next_token)
        return generated.tolist()
    
class CaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, patch_feats, pooled_feats, captions, teacher_forcing=True):
        # Training path: consumes precomputed CLIP features directly (the
        # frozen encoder is bypassed entirely during training).
        return self.decoder(captions, patch_feats, pooled_feats, teacher_forcing=teacher_forcing)

    def generate_from_features(self, patch_feats, pooled_feats, max_len=30, sos_indx=1, eos_indx=2, device='cuda'):
        return self.decoder.generate(patch_feats, pooled_feats, max_len=max_len, sos_indx=sos_indx, eos_indx=eos_indx, device=device)

    def forward_images(self, images, captions, teacher_forcing=True):
        # Convenience path for raw images (e.g. inference on new data).
        patch_feats, pooled_feats = self.encoder(images)
        return self.decoder(captions, patch_feats, pooled_feats, teacher_forcing=teacher_forcing)

    def generate(self, images, max_len=30, sos_indx=1, eos_indx=2, device='cuda'):
        patch_feats, pooled_feats = self.encoder(images)
        gen = self.decoder.generate(patch_feats, pooled_feats, max_len=max_len, sos_indx=sos_indx, eos_indx=eos_indx, device=device)
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
    vocab = Vocab(min_freq=min_freq, max_size=max_size)
    vocab.build_vocab(sents)
    print(f"Vocab size: {len(vocab)}")
    return vocab

def make_dataloaders(images_root, captions_file, vocab, encoder, batch_size=8, subset=None, cache_path='clip_features.pt'):
    items = load_caption_items(images_root, captions_file)
    if subset is not None and subset < len(items):
        random.seed(42)
        items = random.sample(items, subset)

    # One-time (per cache file) encoding of each unique image.
    image_names = sorted({name for name, _ in items})
    patch_feats, pooled_feats, name2indx = precompute_clip_features(
        encoder, images_root, image_names, cache_path
    )
    dataset = CachedFeatureDataset(items, name2indx, patch_feats, pooled_feats, vocab)

    n = len(dataset)
    indxs = list(range(n))
    random.shuffle(indxs)
    split = int(0.9 * n)
    train_indxs, val_indxs = indxs[:split], indxs[split:]
    train_ds = Subset(dataset, train_indxs)
    val_ds = Subset(dataset, val_indxs)
    # num_workers=0 on purpose: __getitem__ is now just an in-RAM tensor
    # lookup (no image decoding), so worker processes would only add IPC
    # overhead and each would hold its own copy of the feature cache.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=CachedFeatureDataset.collate_fn,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=CachedFeatureDataset.collate_fn,
        num_workers=0, pin_memory=True,
    )
    return train_loader, val_loader

def main():
    # Config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device', device)

    # Paths
    data_root = './flickr8k'
    images_dir = 'Images'
    captions_file = 'captions.txt'

    images_root = os.path.join(data_root, images_dir)
    captions_file_path = os.path.join(data_root, captions_file)

    if not os.path.exists(images_root):
        raise FileNotFoundError(f"Images directory not found: {images_root}")
    if not os.path.exists(captions_file_path):
        raise FileNotFoundError(f"Captions file not found: {captions_file_path}")
    
    # Hyperparameters
    epochs = 100
    batch_size = 512
    embed_size = 512
    hidden_size = 1024
    lr = 3e-3
    subset = None
    save_dir = './checkpoints'

    # Build Vocab
    # min_freq=1, max_size=None: every word in the dataset gets a real vocab
    # entry (no UNK collapsing), since the GPU can handle the larger
    # embedding/output layers. The model below sizes itself from len(vocab),
    # so no vocab_size cap is needed.
    vocab = build_vocab_from_captions(captions_file_path, min_freq=1, max_size=None)

    # Model
    encoder = EncoderCLIP(device=device)  # now defaults to ViT-L/14
    decoder = DecoderGRU(len(vocab), feature_dim=encoder.feature_dim, embed_size=embed_size, hidden_size=hidden_size)
    model = CaptioningModel(encoder, decoder).to(device)

    # Dataloaders
    # Dataloaders (features are precomputed once and cached; cache filename
    # is keyed to the CLIP model so switching models triggers a recompute)
    feature_cache = os.path.join(data_root, f'clip_features_{encoder.clip_name}.pt')
    train_loader, val_loader = make_dataloaders(images_root, captions_file_path, vocab, encoder, batch_size=batch_size, subset=subset, cache_path=feature_cache)

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
                    print('Correct Caption :', gt)
                    print('Predicted Caption:', gen)
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