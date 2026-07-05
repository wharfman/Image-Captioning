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
        with open(self.captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    imgcap, caption = line.split('\t')
                except ValueError:
                    continue
                img_name = imgcap.split('#')[0]
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
    def __init__(self, device='gpu'):
        super().__init__()
        self.device = device
        self.model, self.preprocess = clip.load('ViT-B/32', device=device)

        # freeze parameters
        for p in self.model.parameters():
            p.requires_grad = False


    def forward(self, images):
        # images: preprocesssed images tensor (B, 3, H, W)
        with torch.no_gradd():
            img_features = self.model.encode_images(images)

            # normalize vectors to lie on the same scale
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            img_features = img_features.to(torch.float32)

        return img_features