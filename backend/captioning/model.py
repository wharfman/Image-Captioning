import torch
from torch import nn
import torch.nn.functional as F
import clip


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
