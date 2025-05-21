import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, L, d]
        # pe: [max_len, d]
        return x + self.pe[: x.shape[1]]


class FeedForward(nn.Sequential):
    def __init__(self, in_dim, out_dim=None, dropout=None):
        if out_dim is None:
            out_dim = in_dim
        super().__init__(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(out_dim, out_dim),
            nn.LeakyReLU(),
        )


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale=None, attn_mask=None
):
    # q: [B, L, D]
    # k: [B, S, D]
    # v: [B, S, D]
    #
    # w: [B, L, D] @ [B, D, S] + [B, L, S] = [B, L, S]
    # w @ v: [B, L, S] @ [B, S, D] = [B, L, D]
    #
    # mask[:, i, j] - should i attend to j
    L, S = q.shape[-2], k.shape[-2]
    scale_factor = q.shape[-1] ** (-0.5) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=q.dtype, device=q.device)
    if attn_mask is not None:
        attn_bias = torch.zeros_like(attn_mask, dtype=q.dtype, device=q.device)
        attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
    else:
        attn_bias = attn_bias[None, ...]
    w = torch.matmul(q, k.transpose(-1, -2)) * scale_factor
    w += attn_bias
    w = torch.softmax(w, dim=-1)
    return w @ v


class Attention(nn.Module):
    def __init__(self, dim, h_dim):
        super().__init__()

        self.scale = h_dim ** (-0.5)
        self.queries = nn.Linear(dim, h_dim, bias=False)
        self.keys = nn.Linear(dim, h_dim, bias=False)
        self.values = nn.Linear(dim, h_dim, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(h_dim, dim)) if h_dim != dim else nn.Identity()
        )

    def forward(self, x, attn_mask=None):
        # x: [B, L, D]
        # attn_mask: [B, L]
        q = self.queries(x)
        k = self.keys(x)
        v = self.values(x)
        # [B, L, H]

        out = scaled_dot_product_attention(
            q, k, v, scale=self.scale, attn_mask=attn_mask
        )

        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, h_dim):
        super().__init__()
        self.attn = Attention(dim, h_dim)
        self.ffn = FeedForward(dim)
        self.fnorm = nn.LayerNorm(dim)
        self.snorm = nn.LayerNorm(dim)

    def forward(self, x, attn_mask=None):
        y = self.attn(x, attn_mask=attn_mask)
        y = self.fnorm(y + x)
        z = self.ffn(y)
        return self.snorm(z + y)


class Transformer(nn.Module):
    def __init__(self, dim, h_dim, n_layers):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, h_dim) for _ in range(n_layers)]
        )

    def forward(self, x, attn_mask=None):
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        return x
