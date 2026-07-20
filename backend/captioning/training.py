import os
import csv
import random

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from .config import CLIP_MODEL, EMBED_SIZE, HIDDEN_SIZE
from .vocab import Vocab
from .model import EncoderCLIP, DecoderGRU, CaptioningModel
from .datasets import CachedFeatureDataset, load_caption_items, precompute_clip_features


class Trainer:
    def __init__(self, model, vocab, device='cuda', save_dir='./checkpoints'):
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


def main(data_root='./flickr30k', save_dir='./checkpoints'):
    # Config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device', device)

    # Hyperparameters (architecture values come from captioning.config --
    # shared with app.main)
    epochs = 100
    batch_size = 384
    embed_size = EMBED_SIZE
    hidden_size = HIDDEN_SIZE
    lr = 3e-3
    subset = None

    # Encoder first: dataloaders need it to precompute/load features, and
    # the cache filename is keyed to the CLIP model name.
    encoder = EncoderCLIP(device=device, clip_model=CLIP_MODEL)

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
    patience = 5
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
    df = pd.read_csv(log_file)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
