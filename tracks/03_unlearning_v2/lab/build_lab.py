"""Build Ground-Truth Unlearning Lab (Track G) with orig_lab and gold_lab models."""

import argparse
import copy
from pathlib import Path
import torch
import torch.nn.functional as F

from task_template import DEVICE, IMAGE_SIZE, load_model, preprocess


def _load_dataset_tensors(dataset_name: str, max_per_class: int = 200) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load 100-class RGB uint8 tensors [N, 224, 224, 3] for train and val."""
    import torchvision.transforms.functional as TF

    if dataset_name == "cifar100":
        from torchvision.datasets import CIFAR100
        ds_train = ((img, lbl) for img, lbl in CIFAR100(root="dataset", train=True, download=True))
        ds_val = ((img, lbl) for img, lbl in CIFAR100(root="dataset", train=False, download=True))
        rows_tr = ({"image": img, "label": lbl} for img, lbl in ds_train)
        rows_va = ({"image": img, "label": lbl} for img, lbl in ds_val)
    else:
        from datasets import load_dataset
        ds_dict = load_dataset(dataset_name)
        rows_tr = ds_dict["train"]
        rows_va = ds_dict["validation" if "validation" in ds_dict else "test"]

    def _collect(rows, limit_per_cls: int) -> tuple[torch.Tensor, torch.Tensor]:
        counts = [0] * 100
        imgs, labels = [], []
        for row in rows:
            y = int(row["label"] if "label" in row else row["fine_label"])
            if counts[y] >= limit_per_cls:
                continue
            counts[y] += 1
            img_obj = row["image"] if "image" in row else row["img"]
            pil = img_obj.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            imgs.append(TF.pil_to_tensor(pil).permute(1, 2, 0))
            labels.append(y)
            if all(c >= limit_per_cls for c in counts):
                break
        return torch.stack(imgs), torch.tensor(labels, dtype=torch.long)

    tr_x, tr_y = _collect(rows_tr, max_per_class)
    va_x, va_y = _collect(rows_va, 30)
    return tr_x, tr_y, va_x, va_y


def _split_retain_forget(tr_x: torch.Tensor, tr_y: torch.Tensor, seed: int = 42) -> dict[str, torch.Tensor]:
    """Mimic Track 3 structure: 45 active forget classes (~70 imgs/cls) + retain across all 100."""
    gen = torch.Generator().manual_seed(seed)
    forget_classes = set(torch.randperm(100, generator=gen)[:45].tolist())
    r_idx, f_idx = [], []
    for c in range(100):
        idx = torch.where(tr_y == c)[0]
        if c in forget_classes:
            f_idx.append(idx[:70])
            r_idx.append(idx[70:])
        else:
            r_idx.append(idx)
    r_cat, f_cat = torch.cat(r_idx), torch.cat(f_idx)
    return {"r_x": tr_x[r_cat], "r_y": tr_y[r_cat], "f_x": tr_x[f_cat], "f_y": tr_y[f_cat]}


def _train_lab_model(base_model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, epochs: int = 4, batch_size: int = 64) -> torch.nn.Module:
    """Fine-tune a copy of model.pt on (x, y) using identical protocol for orig_lab and gold_lab."""
    model = copy.deepcopy(base_model).to(DEVICE)
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith("blocks.10") or n.startswith("blocks.11") or n.startswith("norm.") or n.startswith("head."))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(x))
        for i in range(0, len(x), batch_size):
            idx = perm[i : i + batch_size]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                logits = model(preprocess(x[idx]))
            loss = F.cross_entropy(logits.float(), y[idx].to(DEVICE))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model.eval()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Ground-Truth Lab (orig_lab + gold_lab).")
    parser.add_argument("--model-path", type=Path, default=Path("model.pt"))
    parser.add_argument("--dataset", type=str, default="clane9/imagenet-100")
    parser.add_argument("--out-dir", type=Path, default=Path("output/lab"))
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = load_model(args.model_path)
    tr_x, tr_y, va_x, va_y = _load_dataset_tensors(args.dataset)
    splits = _split_retain_forget(tr_x, tr_y)
    half = len(va_x) // 2
    splits.update({"ctrl_x": va_x[:half], "ctrl_y": va_y[:half], "val_x": va_x[half:], "val_y": va_y[half:]})

    orig_lab = _train_lab_model(base, torch.cat([splits["r_x"], splits["f_x"]]), torch.cat([splits["r_y"], splits["f_y"]]), epochs=args.epochs)
    gold_lab = _train_lab_model(base, splits["r_x"], splits["r_y"], epochs=args.epochs)

    torch.save({k: v.detach().cpu().float() for k, v in orig_lab.state_dict().items()}, args.out_dir / "orig_lab.pt")
    torch.save({k: v.detach().cpu().float() for k, v in gold_lab.state_dict().items()}, args.out_dir / "gold_lab.pt")
    torch.save(splits, args.out_dir / "lab_splits.pt")
    print(f"Lab saved to {args.out_dir}: retain={len(splits['r_x'])}, forget={len(splits['f_x'])}, val={len(splits['val_x'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
