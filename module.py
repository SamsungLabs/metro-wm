# Adapted from LeWorldModel (LeWM); MIT License, see LICENSES/LeWM-LICENSE.
"""Predictor, action encoder and projector modules of LeWM / PLDM (see jepa.py)."""
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Causal multi-head self-attention."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
                       if project_out else nn.Identity())

    def forward(self, x, causal=True):
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads)
                   for t in self.to_qkv(x).chunk(3, dim=-1))
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        return self.to_out(rearrange(out, "b h t d -> b t (h d)"))


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning on the action embedding."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth, heads, dim_head, mlp_dim,
                 dropout=0.0, block_class=ConditionalBlock):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.cond_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.output_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        self.layers = nn.ModuleList(
            [block_class(hidden_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)])

    def forward(self, x, c=None):
        # the projections are absent from checkpoints saved by older versions of this class
        if hasattr(self, "input_proj"):
            x = self.input_proj(x)
        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)
        for block in self.layers:
            x = block(x, c)
        x = self.norm(x)
        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x


class ARPredictor(nn.Module):
    """Autoregressive latent predictor, causal over frames, conditioned on actions."""

    def __init__(self, *, num_frames, depth, heads, mlp_dim, input_dim, hidden_dim,
                 output_dim=None, dim_head=64, dropout=0.0, emb_dropout=0.0):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(input_dim, hidden_dim, output_dim or input_dim, depth, heads,
                                       dim_head, mlp_dim, dropout, block_class=ConditionalBlock)

    def forward(self, x, c):
        """x: (B, T, D) latents; c: (B, T, D) action embeddings."""
        x = x + self.pos_embedding[:, :x.size(1)]
        return self.transformer(self.dropout(x), c)


class Embedder(nn.Module):
    """Action encoder: one action block (action_dim x frameskip) -> emb_dim."""

    def __init__(self, input_dim=10, smoothed_dim=10, emb_dim=10, mlp_scale=4):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(nn.Linear(smoothed_dim, mlp_scale * emb_dim), nn.SiLU(),
                                   nn.Linear(mlp_scale * emb_dim, emb_dim))

    def forward(self, x):
        x = self.patch_embed(x.float().permute(0, 2, 1)).permute(0, 2, 1)
        return self.embed(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, norm_fn=nn.LayerNorm, act_fn=nn.GELU):
        super().__init__()
        norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), norm, act_fn(),
                                 nn.Linear(hidden_dim, output_dim or input_dim))

    def forward(self, x):
        return self.net(x)
