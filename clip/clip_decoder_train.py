from clip_decoder import Decoder

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from data import build_dataset, get_collate_fn


def main():
    root = Path(__file__).parent.parent

    data = build_dataset()
    collate_fn = get_collate_fn()
    loader = torch.utils.data.DataLoader(data, 16, collate_fn=collate_fn, num_workers=8)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    n_tokens = 20
    model = Decoder(root / "clip.pt", n_tokens=n_tokens).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    def epoch(model, optim, loader):
        losses = list()
        pbar = tqdm(loader)
        for batch in pbar:
            optim.zero_grad()
            images = batch["images"].to(device)
            input_ids = batch["input_ids"][:, :-1, ...].to(device)
            attention_mask = batch["attention_mask"][:, :-1, ...].to(device)
            targets = batch["input_ids"].to(device)
            logits = (
                model(images, input_ids, attention_mask)
                .logits[:, n_tokens - 1 :, ...]
                .transpose(-2, -1)
            )
            loss = F.cross_entropy(logits, targets)
            loss_item = loss.detach().cpu().item()
            pbar.set_description(f"loss={loss_item}")
            losses.append(loss_item)
            loss.backward()
            optim.step()
        return np.mean(losses)

    for i in range(10):
        loss = epoch(model, optim, loader)
        print(f"epoch {i}: loss={loss}")
        torch.save(model.state_dict(), root / "decoder.pt")


if __name__ == "__main__":
    main()
