'''
Training entrypoint.

    python train.py

Run from anywhere -- paths are resolved relative to the repo root regardless
of the current working directory, so `python backend/train.py` from the repo
root and `python train.py` from inside backend/ both work identically.

Edit hyperparameters at the top of captioning.training.main() to tune a run.
'''
from pathlib import Path

from captioning.training import main

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
DATA_ROOT = BASE_DIR / 'flickr30k'
SAVE_DIR = BASE_DIR / 'checkpoints'

if __name__ == '__main__':
    main(data_root=str(DATA_ROOT), save_dir=str(SAVE_DIR))
