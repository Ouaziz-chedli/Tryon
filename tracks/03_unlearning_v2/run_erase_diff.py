"""Escanade: weight-difference map between an extreme unlearning checkpoint and orig model.pt.

Locates where instance memory lives: per-tensor deltas, group aggregation
(patch_embed / pos_embed / cls_token / blocks.N subgroups / head), per-row
head deltas on active vs absent classes, and top-1 SVD of late-block mlp.fc2
deltas. Pure CPU tensor math; the only forward pass is the orig-model pass
used to derive the 45 active classes (repo convention from eval_harness).
"""

import argparse
import json
from pathlib import Path

import torch

from task_template import NUM_CLASSES, load_forget_images, load_model, predict

DEFAULT_EXTREME = Path("output/track_i/I2R_f6_r4_ce18_fl085.pt")
# methods/common.restore_absent_head_rows copies absent rows verbatim, so any
# nonzero delta there means the checkpoint pipeline leaked; tolerate fp noise only.
ABSENT_ROW_TOL = 1e-5
SVD_BLOCK_RANGE = range(6, 12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map weight deltas between orig and an extreme unlearned checkpoint.")
    parser.add_argument("--orig", type=Path, default=Path("model.pt"))
    parser.add_argument("--extreme", type=Path, default=DEFAULT_EXTREME)
    parser.add_argument("--forget-path", type=Path, default=Path("forget_images.npz"))
    parser.add_argument("--output", type=Path, default=Path("output/erase_diff/erase_diff_report.json"))
    parser.add_argument("--max-images", type=int, default=0, help="0 = all forget images (active-class mask).")
    return parser.parse_args()


def load_state_dict_cpu(path: Path) -> dict[str, torch.Tensor]:
    """Load a strict-repo state dict on CPU (same call as eval_harness.validate_checkpoint)."""
    return torch.load(path, map_location="cpu", weights_only=True)


def verify_matching_keys(orig_sd: dict, extreme_sd: dict, orig: Path, extreme: Path) -> None:
    missing = sorted(set(orig_sd) - set(extreme_sd))
    if missing or set(orig_sd) != set(extreme_sd):
        raise KeyError(f"state dict keys differ between {orig} and {extreme}; missing in extreme: {missing[:5]}")
    mismatched = [k for k in orig_sd if orig_sd[k].shape != extreme_sd[k].shape]
    if mismatched:
        raise ValueError(f"shape mismatch for {mismatched[:5]}")


def _strip_param_suffix(sub: str) -> str:
    for suffix in (".weight", ".bias", ".gamma"):
        if sub.endswith(suffix):
            return sub[: -len(suffix)]
    return sub


def split_group(name: str) -> tuple[str, str]:
    """Map a state-dict key to (group, subgroup); head.weight/head.bias stay whole."""
    if name in ("pos_embed", "cls_token") or name.startswith("head."):
        return name, ""
    if name.startswith("patch_embed."):
        return "patch_embed", _strip_param_suffix(name[len("patch_embed."):])
    if name.startswith("blocks."):
        _, idx, rest = name.split(".", 2)
        return f"blocks.{idx}", _strip_param_suffix(rest)
    return "other", name


@torch.no_grad()
def compute_active_mask(orig_path: Path, forget_path: Path, max_images: int) -> torch.Tensor:
    """Active classes = classes orig predicts on forget images (eval_harness convention)."""
    images = load_forget_images(forget_path)
    if max_images > 0:
        images = images[:max_images]
    counts = torch.bincount(predict(load_model(orig_path), images).argmax(dim=-1), minlength=NUM_CLASSES)
    return counts > 0


def tensor_delta_norms(orig_t: torch.Tensor, sub_t: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Delta tensor plus its l2 / relative-l2 / linf norms."""
    delta = sub_t.float() - orig_t.float()
    l2 = float(delta.norm())
    orig_l2 = float(orig_t.float().norm())
    if orig_l2 > 0:
        rel = l2 / orig_l2
    else:
        rel = float("inf") if l2 > 0 else 0.0
    linf = float(delta.abs().max()) if delta.numel() else 0.0
    return delta, {"l2": round(l2, 6), "rel": round(rel, 8), "lInf": round(linf, 8)}


def _accumulate(sums: dict[str, list[float]], key: str, delta: torch.Tensor, orig_l2: float) -> None:
    entry = sums.setdefault(key, [0.0, 0.0])
    entry[0] += float(delta.pow(2).sum())
    entry[1] += orig_l2 ** 2


def _rel_norm(delta_l2: float, orig_l2: float) -> float:
    if orig_l2 > 0:
        return delta_l2 / orig_l2
    return float("inf") if delta_l2 > 0 else 0.0


def finalize_ranked(sums: dict[str, list[float]]) -> list[dict]:
    """Turn {key: [delta_sq, orig_sq]} sums into entries sorted by relative norm desc."""
    entries = []
    for key, (delta_sq, orig_sq) in sums.items():
        l2, orig_l2 = delta_sq ** 0.5, orig_sq ** 0.5
        entries.append({"group": key, "l2": round(l2, 6), "rel": round(_rel_norm(l2, orig_l2), 8)})
    return sorted(entries, key=lambda e: -e["rel"])


def analyze_head(delta_w: torch.Tensor, delta_b: torch.Tensor, active_mask: torch.Tensor) -> dict:
    """Per-row head deltas: instance memory carriers vs the 55 absent rows (must be ~0)."""
    absent = ~active_mask
    absent_max = float(delta_w[absent].abs().max()) if absent.any() else 0.0
    absent_bias_max = float(delta_b[absent].abs().max()) if absent.any() else 0.0
    row_l2 = delta_w[active_mask].norm(dim=1)
    return {
        "num_active": int(active_mask.sum()),
        "num_absent": int(absent.sum()),
        "active_row_l2_mean": round(float(row_l2.mean()), 6),
        "active_row_l2_max": round(float(row_l2.max()), 6),
        "absent_weight_max_abs": round(absent_max, 8),
        "absent_bias_max_abs": round(absent_bias_max, 8),
        "absent_integrity_ok": bool(max(absent_max, absent_bias_max) <= ABSENT_ROW_TOL),
    }


def top_singular_stats(delta_2d: torch.Tensor) -> dict:
    """Top-1 SVD of a delta matrix: dominant direction weight and its energy share."""
    sv = torch.linalg.svdvals(delta_2d.float())
    top = float(sv[0])
    energy = top ** 2 / float((sv.float() ** 2).sum())
    return {"top_sv": round(top, 6), "energy_frac": round(energy, 4)}


def scan_tensor_deltas(
    orig_sd: dict[str, torch.Tensor],
    extreme_sd: dict[str, torch.Tensor],
) -> tuple[list[dict], dict[str, list[float]], dict[str, list[float]]]:
    """Per-tensor delta norms plus squared-delta/orig accumulators per group and subgroup."""
    tensors, group_sums, sub_sums = [], {}, {}
    for name, orig_t in orig_sd.items():
        delta, norms = tensor_delta_norms(orig_t, extreme_sd[name])
        orig_l2 = float(orig_t.float().norm())
        tensors.append({"name": name, "shape": list(orig_t.shape), **norms})
        group, subgroup = split_group(name)
        _accumulate(group_sums, group, delta, orig_l2)
        if subgroup:
            _accumulate(sub_sums, f"{group}.{subgroup}", delta, orig_l2)
    return tensors, group_sums, sub_sums


def build_report(
    orig_sd: dict[str, torch.Tensor],
    extreme_sd: dict[str, torch.Tensor],
    active_mask: torch.Tensor,
    orig: Path,
    extreme: Path,
) -> dict:
    tensors, group_sums, sub_sums = scan_tensor_deltas(orig_sd, extreme_sd)
    return {
        "orig": str(orig),
        "extreme": str(extreme),
        "num_tensors": len(tensors),
        "total_delta_l2": round(sum(t["l2"] ** 2 for t in tensors) ** 0.5, 6),
        "head_integrity": analyze_head(
            extreme_sd["head.weight"] - orig_sd["head.weight"],
            extreme_sd["head.bias"] - orig_sd["head.bias"],
            active_mask,
        ),
        "tensors": tensors,
        "top_tensors": sorted(tensors, key=lambda t: -t["rel"])[:20],
        "groups": finalize_ranked(group_sums),
        "block_subgroups": finalize_ranked(sub_sums),
        "svd_mlp_fc2": {
            f"blocks.{i}": top_singular_stats(extreme_sd[f"blocks.{i}.mlp.fc2.weight"].float()
                                              - orig_sd[f"blocks.{i}.mlp.fc2.weight"].float())
            for i in SVD_BLOCK_RANGE
        },
    }


def print_summary(report: dict) -> None:
    print(f"erase diff: {report['extreme']} vs {report['orig']}")
    print(f"total delta L2 = {report['total_delta_l2']}")
    head = report["head_integrity"]
    verdict = "OK" if head["absent_integrity_ok"] else f"WARNING: leak {head['absent_weight_max_abs']}"
    print(f"head integrity: absent rows delta max = {head['absent_weight_max_abs']} -> {verdict}")
    print(f"active rows: mean L2 = {head['active_row_l2_mean']}, max L2 = {head['active_row_l2_max']}")
    print("\ntop groups by relative delta norm:")
    for entry in report["groups"][:12]:
        print(f"  {entry['group']:<24} rel={entry['rel']:.4%}  l2={entry['l2']}")
    print("\ntop-20 tensors by relative delta norm:")
    for entry in report["top_tensors"]:
        print(f"  {entry['name']:<36} rel={entry['rel']:.4%}  l2={entry['l2']}  linf={entry['lInf']}")
    print("\ntop-1 SVD of late-block mlp.fc2 deltas:")
    for block, stats in report["svd_mlp_fc2"].items():
        print(f"  {block}: top_sv={stats['top_sv']}  energy={stats['energy_frac']:.1%}")


def main() -> int:
    args = parse_args()
    for path in (args.orig, args.extreme, args.forget_path):
        if not path.exists():
            raise FileNotFoundError(f"missing input: {path}")
    orig_sd = load_state_dict_cpu(args.orig)
    extreme_sd = load_state_dict_cpu(args.extreme)
    verify_matching_keys(orig_sd, extreme_sd, args.orig, args.extreme)
    active_mask = compute_active_mask(args.orig, args.forget_path, args.max_images)
    report = build_report(orig_sd, extreme_sd, active_mask, args.orig, args.extreme)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print_summary(report)
    print(f"\nreport written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
