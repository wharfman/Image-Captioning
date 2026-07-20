'''
Model architecture configuration -- single source of truth.

app.main imports these, so changing them here is the ONLY place they need to
change. Note: they describe the architecture a checkpoint was trained with;
if you change them, existing checkpoints will no longer load until you
retrain (the weight shapes must match).
'''

CLIP_MODEL = 'ViT-L/14'
EMBED_SIZE = 512
HIDDEN_SIZE = 1024
MAX_LEN = 30  # max generated caption length at inference
