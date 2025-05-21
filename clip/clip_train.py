from clip import CLIPModel

from torchvision.datasets import CocoCaptions
from transformers import AutoTokenizer
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from data import build_dataset, get_collate_fn


def main():
    root = Path(__file__).parent.parent

    data = build_dataset()
    collate_fn = get_collate_fn()
    loader = torch.utils.data.DataLoader(data, 16, collate_fn=collate_fn, num_workers=8)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = CLIPModel().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    def epoch(model, optim, loader):
        losses = list()
        pbar = tqdm(loader)
        for batch in pbar:
            optim.zero_grad()
            loss, _ = model(
                batch["images"].to(device),
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            loss_item = loss.detach().cpu().item()
            pbar.set_description(f"loss={loss_item}")
            losses.append(loss_item)
            loss.backward()
            optim.step()
        return np.mean(losses)

    for i in range(10):
        loss = epoch(model, optim, loader)
        print(f"epoch {i}: loss={loss}")
        torch.save(model.state_dict(), root / "clip.pt")


if __name__ == "__main__":
    main()
