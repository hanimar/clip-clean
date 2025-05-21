from transformer import Transformer, PositionalEmbedding

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


class ImageEncoder(nn.Module):
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


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50257, 256)
        self.pe = PositionalEmbedding(256, 1024)
        self.transformer = Transformer(256, 256, 8)

    def forward(self, input_ids, attention_mask=None):
        embs = self.emb(input_ids)
        embs = self.pe(embs)
        # attention_mask: [B, L] | 0=pad
        attn_mask = attention_mask[:, None, :] if attention_mask is not None else None
        return self.transformer(embs, attn_mask=attn_mask)


class CLIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()

    def encode_image(self, image):
        enc = self.img_encoder(image)
        return enc.mean(dim=-2)

    def encode_text(self, input_ids, attention_mask):
        enc = self.text_encoder(input_ids, attention_mask)
        denom = attention_mask.sum(dim=-1, keepdim=True)
        return torch.sum(enc * attention_mask[..., None], dim=-2) / denom

    def forward(self, images, input_ids, attention_mask):
        i_emb = self.encode_image(images)
        t_emb = self.encode_text(input_ids, attention_mask)
        logits = i_emb @ t_emb.T
        labels = torch.arange(i_emb.shape[0], device=i_emb.device)
        l1 = F.cross_entropy(logits, labels)
        l2 = F.cross_entropy(logits.T, labels)
        loss = (l1 + l2) * 0.5
        return loss, logits
