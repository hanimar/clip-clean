import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from clip import CLIPModel


class Projector(nn.Module):
    def __init__(self, n_tokens=20):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(256, 384 * n_tokens),
            nn.LeakyReLU(),
            nn.Linear(384 * n_tokens, 768 * n_tokens),
        )
        self.n_tokens = n_tokens

    def forward(self, image_features):
        t = self.model(image_features)
        return t.reshape(-1, self.n_tokens, 768)


class Decoder(nn.Module):
    def __init__(self, path_to_clip, n_tokens=20):
        super().__init__()
        self.clip = CLIPModel()
        self.clip.load_state_dict(torch.load(path_to_clip))
        self.clip.eval()
        for param in self.clip.parameters():
            param.requires_grad = False

        self.projector = Projector(n_tokens)

        self.language_model = AutoModelForCausalLM.from_pretrained("gpt2")
        for param in self.language_model.parameters():
            param.requires_grad = False

        for param in self.language_model.transformer.h[0].parameters():
            param.requires_grad = True

    def forward(self, images, input_ids, attention_mask):
        image_features = self.clip.encode_image(images)
        image_embeds = self.projector(image_features)
        text_embeds = self.language_model.transformer.wte(input_ids)
        embeds_cat = torch.cat([image_embeds, text_embeds], dim=1)
        t_mask = torch.ones(
            attention_mask.shape[0],
            self.projector.n_tokens,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        mask_cat = torch.cat([t_mask, attention_mask], dim=1)
        captions = self.language_model(
            inputs_embeds=embeds_cat, attention_mask=mask_cat
        )
        return captions

    def forward_single(self, image, input_ids):
        images = image[None, ...]
        input_ids = input_ids[None, ...]
        attention_mask = torch.ones_like(input_ids)
        out = self.forward(images, input_ids, attention_mask)
        return out
