from model import Img2TextTransformer

from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from data import build_dataset, get_collate_fn
import torch.nn.functional as F


def main():
    root = Path(__file__).parent.parent

    data = build_dataset()
    collate_fn = get_collate_fn()
    loader = torch.utils.data.DataLoader(data, 16, collate_fn=collate_fn, num_workers=8)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = Img2TextTransformer().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    def epoch(model, optim, loader):
        losses = list()
        pbar = tqdm(loader)
        for batch in pbar:
            optim.zero_grad()
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(images, input_ids[:, :-1], attention_mask[:, :-1]).transpose(
                -2, -1
            )
            targets = input_ids[:, 1:]
            loss = (
                F.cross_entropy(logits, targets, reduction="none")
                * attention_mask[:, 1:]
            ).mean()
            loss_item = loss.detach().cpu().item()
            pbar.set_description(f"loss={loss_item}")
            losses.append(loss_item)
            loss.backward()
            optim.step()
        return np.mean(losses)

    for i in range(10):
        loss = epoch(model, optim, loader)
        print(f"epoch {i}: loss={loss}")
        torch.save(model.state_dict(), root / "transformer.pt")


if __name__ == "__main__":
    main()
