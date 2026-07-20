import os
import csv

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm


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
