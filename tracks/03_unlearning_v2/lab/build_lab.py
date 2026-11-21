"""Build Ground-Truth Unlearning Lab (Track G) with orig_lab and gold_lab models."""

import argparse
import copy
from pathlib import Path
import torch
import torch.nn.functional as F

from task_template import DEVICE, IMAGE_SIZE, build_model, load_model, preprocess


def _load_dataset_tensors(dataset_name: str, max_per_class: int = 350) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    va_x, va_y = _collect(rows_va, 50)
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


def _lab_param_groups(model: torch.nn.Module) -> list[dict]:
    """Use higher LR for head to realign dataset label indices, lower LR for ViT backbone."""
    head = [p for n, p in model.named_parameters() if "head" in n]
    backbone = [p for n, p in model.named_parameters() if "head" not in n]
    groups = [{"params": head, "lr": 1e-3}, {"params": backbone, "lr": 5e-5}]
    return [g for g in groups if g["params"]]


def _train_lab_model(base_model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, epochs: int = 10, batch_size: int = 128) -> torch.nn.Module:
    """Fine-tune a copy of model.pt on (x, y) using identical protocol for orig_lab and gold_lab."""
    import torchvision.transforms as T

    aug = T.Compose([T.RandomHorizontalFlip(p=0.5), T.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0), antialias=True)])
    model = copy.deepcopy(base_model).to(DEVICE).requires_grad_(True)
    opt = torch.optim.AdamW(_lab_param_groups(model), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(x))
        tot_loss, tot_ok = 0.0, 0
        for i in range(0, len(x), batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = aug(preprocess(x[idx])), y[idx].to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                logits = model(xb)
            loss = F.cross_entropy(logits.float(), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_loss += float(loss.detach()) * len(idx)
            tot_ok += int((logits.argmax(dim=-1) == yb).sum())
        sched.step()
        print(f"  [lab train] epoch {ep + 1}/{epochs} loss={tot_loss / len(x):.4f} acc={tot_ok / len(x):.4f}", flush=True)
    return model.eval()


def _load_lab_base(init_mode: str, model_path: Path) -> torch.nn.Module:
    """Load Lab base ViT-B/16: ImageNet-1k pretrained (with LayerScale=1.0) or model.pt."""
    if init_mode == "model":
        return load_model(model_path)
    import timm

    model = build_model()
    src = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0).state_dict()
    model.load_state_dict(src, strict=False)
    for name, param in model.named_parameters():
        if name.endswith(".gamma"):
            param.data.fill_(1.0)
    return model.to(DEVICE).eval()


def _load_or_build_splits(out_dir: Path, dataset: str, rebuild: bool) -> dict[str, torch.Tensor]:
    """Reuse cached lab_splits.pt when present, ensuring ctrl/val are class-shuffled."""
    cache = out_dir / "lab_splits.pt"
    if cache.exists() and not rebuild:
        splits = torch.load(cache, map_location="cpu", weights_only=True)
        va_x = torch.cat([splits["ctrl_x"], splits["val_x"]])
        va_y = torch.cat([splits["ctrl_y"], splits["val_y"]])
    else:
        tr_x, tr_y, va_x, va_y = _load_dataset_tensors(dataset)
        splits = _split_retain_forget(tr_x, tr_y)
    perm = torch.randperm(len(va_x), generator=torch.Generator().manual_seed(42))
    va_x, va_y = va_x[perm], va_y[perm]
    half = len(va_x) // 2
    splits.update({"ctrl_x": va_x[:half], "ctrl_y": va_y[:half], "val_x": va_x[half:], "val_y": va_y[half:]})
    torch.save(splits, cache)
    return splits


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Ground-Truth Lab (orig_lab + gold_lab).")
    parser.add_argument("--model-path", type=Path, default=Path("model.pt"))
    parser.add_argument("--dataset", type=str, default="clane9/imagenet-100")
    parser.add_argument("--out-dir", type=Path, default=Path("output/lab"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--init-mode", choices=["pretrained", "model"], default="pretrained")
    parser.add_argument("--rebuild-splits", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = _load_lab_base(args.init_mode, args.model_path)
    splits = _load_or_build_splits(args.out_dir, args.dataset, args.rebuild_splits)

    orig_lab = _train_lab_model(base, torch.cat([splits["r_x"], splits["f_x"]]), torch.cat([splits["r_y"], splits["f_y"]]), epochs=args.epochs)
    gold_lab = _train_lab_model(base, splits["r_x"], splits["r_y"], epochs=args.epochs)

    torch.save({k: v.detach().cpu().float() for k, v in orig_lab.state_dict().items()}, args.out_dir / "orig_lab.pt")
    torch.save({k: v.detach().cpu().float() for k, v in gold_lab.state_dict().items()}, args.out_dir / "gold_lab.pt")
    print(f"Lab saved to {args.out_dir}: retain={len(splits['r_x'])}, forget={len(splits['f_x'])}, val={len(splits['val_x'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
