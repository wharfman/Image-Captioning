import os   
import csv
import random
from collections import Counter

import torch
from torch import nn
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
    
class EncoderCLIP(nn.Module):
    def __init__(self, device='cude'):
        super().__init__()
        self.device = device
        self.model, self.preprocess = clip.load('ViT-B/32', device=device)

        # freeze parameters
        for p in self.model.parameters():
            p.requires_grad = False


    def forward(self, images):
        # images: preprocesssed images tensor (B, 3, H, W)
        with torch.no_grad():
            img_features = self.model.encode_image(images)

            # normalize vectors to lie on the same scale
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            img_features = img_features.to(torch.float32)

        return img_features
    
class DecoderGRU(nn.Module):
    def __init__(self, vocab_size, embed_size=300, hidden_size=512, num_layers=1, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.img2hidden = nn.Linear(512, hidden_size)

        # Stack of GRUCells replaces nn.GRU. The fused nn.GRU kernel hits
        # MIOpen's RNN descriptor path, which crashes with
        # miopenStatusUnknownError on some ROCm setups. GRUCell instead uses
        # plain matmul/elementwise ops, which MIOpen handles without issue.
        self.cells = nn.ModuleList([
            nn.GRUCell(embed_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        # Dropout between stacked layers only, matching nn.GRU's own convention
        # (no dropout after the final layer's output).
        self.dropout = nn.Dropout(dropout) if num_layers > 1 else nn.Identity()
        self.fc = nn.Linear(hidden_size, vocab_size)

    def _init_hidden(self, img_features):
        # Same initial hidden state broadcast to every layer, matching the
        # old h0.repeat(num_layers, 1, 1) behavior.
        h0 = torch.tanh(self.img2hidden(img_features)) # (B, H)
        return [h0.clone() for _ in range(self.num_layers)]

    def _step(self, input_t, hidden_states):
        # input_t: (B, E) ; hidden_states: list of (B, H), one per layer
        x = input_t
        new_hidden = []
        for i, cell in enumerate(self.cells):
            h = cell(x, hidden_states[i])
            new_hidden.append(h)
            x = h
            if i < self.num_layers - 1:
                x = self.dropout(x)
        return x, new_hidden # x is the output of the last layer, (B, H)

    def forward(self, captions, img_features, lengths=None, teacher_forcing=True):
        # captions: (B, T) token ids including SOS at pos 0
        # img_features: (B, 512)
        B, T = captions.size()
        device = captions.device
        embeddings = self.embedding(captions) # (B, T, E)

        hidden_states = self._init_hidden(img_features)
        outputs = torch.zeros(B, T, self.vocab_size, device=device)

        if teacher_forcing:
            # step through the sequence manually (was a single fused GRU call)
            for t in range(T - 1):
                input_t = embeddings[:, t, :] # (B, E)
                out, hidden_states = self._step(input_t, hidden_states)
                logits = self.fc(out) # (B, V)
                outputs[:, t + 1, :] = logits
            return outputs
        else:
            # step-by-step generation
            input_t = embeddings[:, 0, :] # SOS embedding, (B, E)
            for t in range(1, T):
                out, hidden_states = self._step(input_t, hidden_states)
                logit = self.fc(out) # (B, V)
                outputs[:, t, :] = logit

                next_token = logit.argmax(dim=-1)
                input_t = self.embedding(next_token)
            return outputs
        
    def generate(self, img_features, max_len=30, sos_indx=1, eos_indx=2, device='cude'):
        # greedy decoding for a single image/batch
        if img_features.dim() == 1:
            img_features = img_features.unsqueeze(0)
        B = img_features.size(0)
        generated = torch.full((B, max_len), fill_value=0, dtype=torch.long, device=device)
        hidden_states = self._init_hidden(img_features)
        input_t = self.embedding(torch.tensor([sos_indx]*B, device=device)) # (B, E)

        for t in range(1, max_len):
            out, hidden_states = self._step(input_t, hidden_states)
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

    def forward(self, images, captions, teacher_forcing=True):
        img_feats = self.encoder(images)
        outputs = self.decoder(captions, img_feats, teacher_forcing=teacher_forcing)
        return outputs
    
    def generate(self, images, max_len=30, sos_indx=1, eos_indx=2, device='cude'):
        img_feats = self.encoder(images)
        gen = self.decoder.generate(img_feats, max_len=max_len, sos_indx=sos_indx, eos_indx=eos_indx, device=device)
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
        pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")

        scaler = torch.amp.GradScaler(device=self.device if self.device in ('cuda', 'cpu') else 'cuda')

        for images, captions, lengths in pbar:
            images = images.to(self.device)
            captions = captions.to(self.device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device if self.device in ('cuda', 'cpu') else 'cuda'):
                outputs = self.model(images, captions, teacher_forcing=True)
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
        for images, captions, lengths in tqdm(dataloader, desc='Valid'):
            images = images.to(self.device)
            captions = captions.to(self.device)
            outputs = self.model(images, captions, teacher_forcing=True)
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

def build_vocab_from_captions(captions_file, min_freq=2, max_size=10000):
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

def make_dataloaders(images_root, captions_file, vocab, encoder, batch_size=8, subset=None, num_workers=8):
    preprocess = encoder.preprocess
    dataset = Flickr8kDataset(images_root, captions_file, vocab, clip_preprocess=preprocess, subset=subset)

    n = len(dataset)
    indxs = list(range(n))
    random.shuffle(indxs)
    split = int(0.9 * n)
    train_indxs, val_indxs = indxs[:split], indxs[split:]
    train_ds = Subset(dataset, train_indxs)
    val_ds = Subset(dataset, val_indxs)
    # num_workers>0 loads/decodes images in parallel worker processes instead
    # of blocking the main thread; pin_memory speeds up the host->GPU copy;
    # persistent_workers avoids re-spawning workers every epoch.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=Flickr8kDataset.collate_fn,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=Flickr8kDataset.collate_fn,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0),
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
    batch_size = 768
    embed_size = 512
    hidden_size = 1024
    lr = 3e-3
    vocab_size = 8000
    subset = None
    save_dir = './checkpoints'

    # Build Vocab
    vocab = build_vocab_from_captions(captions_file_path, min_freq=2, max_size=vocab_size)

    # Model
    encoder = EncoderCLIP(device=device)
    decoder = DecoderGRU(len(vocab), embed_size=embed_size, hidden_size=hidden_size)
    model = CaptioningModel(encoder, decoder).to(device)

    # Dataloaders
    train_loader, val_loader = make_dataloaders(images_root, captions_file_path, vocab, encoder, batch_size=batch_size, subset=subset, num_workers=8)

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
            for images, captions, lengths in val_loader:
                images = images.to(device)
                gen_ids = model.generate(images[:4], max_len=20,
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