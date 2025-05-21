from transformer import Transformer, CrossTransformer, PositionalEmbedding

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )


class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Sequential(
            ConvBnRelu(3, 16, kernel_size=2, stride=2),
            ConvBnRelu(16, 32, kernel_size=2, stride=2),
            ConvBnRelu(32, 64, kernel_size=2, stride=2),
            ConvBnRelu(64, 128, kernel_size=2, stride=2),
            ConvBnRelu(128, 256, kernel_size=2, stride=2),
            nn.Flatten(start_dim=2),
        )
        self.pe = PositionalEmbedding(256, 49)
        self.transformer = Transformer(256, 256, 8)

    def forward(self, x: torch.Tensor):
        # x: [B, 3, 224, 224] | 224 = 2 * 2 * 2 * 2 * 2 * 7
        embs = self.emb(x)  # [B, 128, 49]
        embs = embs.transpose(-1, -2)  # [B, 49, 256]
        embs = self.pe(embs)
        return self.transformer(embs)


class TextTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50257 + 1, 256)
        self.pe = PositionalEmbedding(256, 1024)
        self.transformer = CrossTransformer(256, 256, 8)
        self.proj = nn.Linear(256, 50257 + 1)

    def forward(self, image_tokens, input_ids, attention_mask):
        embs = self.emb(input_ids)
        embs = self.pe(embs)
        # attention_mask: [B, L] | 0=pad
        L = attention_mask.shape[-1]
        attn_mask = attention_mask[:, None, :].repeat(1, L, 1).tril()
        t = self.transformer(embs, image_tokens, attn_mask=attn_mask)
        return self.proj(t)


class Img2TextTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VisionTransformer()
        self.decoder = TextTransformer()

    def forward(self, images, input_ids, attention_mask):
        image_tokens = self.encoder(images)
        return self.decoder(image_tokens, input_ids, attention_mask)

    def forward_single(self, image, input_ids):
        images = image[None, ...]
        input_ids = input_ids[None, ...]
        attention_mask = torch.ones_like(input_ids)
        return self.forward(images, input_ids, attention_mask)
