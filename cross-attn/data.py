from torchvision.datasets import CocoCaptions
from transformers import AutoTokenizer
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
import torch


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    bos_token = "<|startoftext|>"
    tokenizer.add_special_tokens({"additional_special_tokens": [bos_token]})
    tokenizer.bos_token = bos_token
    return tokenizer


def build_dataset():
    root = Path(__file__).parent.parent

    general_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ]
    )

    def target_transform(target):
        return np.random.choice(target).item()

    data = CocoCaptions(
        root=root / "data/train2014",
        annFile=root / "data/annotations/captions_train2014.json",
        transform=general_transforms,
        target_transform=target_transform,
    )
    return data


def get_collate_fn():
    tokenizer = get_tokenizer()

    def collate_fn(x):
        images = torch.stack([t[0] for t in x])
        texts = [tokenizer.bos_token + t[1] + tokenizer.eos_token for t in x]
        proc_texts = tokenizer(texts, padding=True, return_tensors="pt")
        res = {"images": images}
        res.update(proc_texts)
        return res

    return collate_fn
