import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import GPTConfig
from .token_embedding import TokenEmbedding
from .positional_embedding import PositionalEmbedding
from .block import Block


class MiniGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = TokenEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos = PositionalEmbedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok.emb.weight

    def forward(self, idx):
        assert idx.size(1) <= self.cfg.block_size, "sequence too long"
        x = self.drop(self.pos(self.tok(idx)))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def compute_loss(self, idx):
        """
        Next-token language modeling loss (CrossEntropy).
        We train to predict token t+1 z danych do t.
        """
        logits = self(idx)                  # (B, T, V)
        # shift: porównujemy do przyszłości
        targets = idx[:, 1:].contiguous()   # (B, T-1)
        logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)

        B, Tm1, V = logits.shape
        logits = logits.view(B * Tm1, V)
        targets = targets.view(B * Tm1)
        return F.cross_entropy(logits, targets)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, top_k=None, top_p=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits = self(idx_cond)[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            if top_k is not None:
                v, _ = torch.topk(probs, top_k)
                cutoff = v[:, -1].unsqueeze(1)
                probs = torch.where(probs < cutoff, torch.zeros_like(probs), probs)
                probs = probs / probs.sum(dim=-1, keepdim=True)

            if top_p is not None:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum > top_p
                mask[:, 0] = False
                sorted_probs[mask] = 0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                # next_token = torch.multinomial(sorted_probs, 1)
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
                next_token = sorted_idx.gather(1, next_token)
            else:
                next_token = torch.multinomial(probs, 1)

            idx = torch.cat([idx, next_token], dim=1)
        return idx
