"""Aggregate v10 sweep results into a printable table.

Reads:
  - outputs/bc_v10_resnet/<tag>/checkpoint.pth → last val MSE from history or args
  - visualizations/bc_v10_resnet/<tag>/summary.json → AR metrics

Prints a markdown table to stdout (to be pasted into EXPERIMENTS.md).
"""
import json
import os
from pathlib import Path

import torch

PROJ_ROOT = Path("/home/yxy/VQ-VLA")
OUT_ROOT = PROJ_ROOT / "outputs/bc_v10_resnet"
VIZ_ROOT = PROJ_ROOT / "visualizations/bc_v10_resnet"

TAGS = [
    "mlp_trainable", "mlp_frozen",
    "linear64_trainable", "linear64_frozen",
    "raw_trainable", "raw_frozen",
]


def parse_final_val(train_log: Path):
    """Parse the last 'Final val:' line for TF arm / hand / no_corr MSE."""
    if not train_log.exists():
        return None
    text = train_log.read_text().splitlines()
    for line in reversed(text):
        if line.startswith("Final val:"):
            # "Final val: arm=0.034533  hand=0.002961  no_corr=0.002893  vision_gain=-0.000069"
            parts = line.split()
            d = {}
            for p in parts:
                if "=" in p:
                    k, v = p.split("=")
                    try:
                        d[k] = float(v)
                    except ValueError:
                        pass
            return d
    return None


def load_summary_json(viz_dir: Path):
    p = viz_dir / "summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def count_params(ckpt_path: Path):
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return sum(v.numel() for v in ckpt["model"].values())


def main():
    rows = []
    for tag in TAGS:
        outdir = OUT_ROOT / tag
        vizdir = VIZ_ROOT / tag
        tf = parse_final_val(outdir / "train.log") or {}
        ar = load_summary_json(vizdir) or {}
        n_params = count_params(outdir / "checkpoint.pth")
        rows.append({
            "tag": tag,
            "tf_arm": tf.get("arm"),
            "tf_hand": tf.get("hand"),
            "no_corr": tf.get("no_corr"),
            "vision_gain": tf.get("vision_gain"),
            "ar_arm": ar.get("ar_arm_mse"),
            "ar_hand": ar.get("ar_hand_mse"),
            "nc_ar_hand": ar.get("nc_hand_mse"),
            "params": n_params,
        })

    # Markdown table
    print("| tag | TF arm | TF hand | TF no_corr | TF vision_gain | AR arm | AR hand | AR no_corr | params |")
    print("|-----|--------|---------|-----------|----------------|--------|---------|-----------|--------|")
    for r in rows:
        def fmt(v, fmt_spec=".5f"):
            if v is None:
                return "—"
            return f"{v:{fmt_spec}}"
        params = f"{r['params']:,}" if r["params"] is not None else "—"
        print(
            f"| {r['tag']} | {fmt(r['tf_arm'])} | {fmt(r['tf_hand'])} | "
            f"{fmt(r['no_corr'])} | {fmt(r['vision_gain'], '+.5f')} | "
            f"{fmt(r['ar_arm'])} | {fmt(r['ar_hand'])} | {fmt(r['nc_ar_hand'])} | {params} |"
        )

    # Quick best pickers
    def best(key, lower_is_better=True):
        valid = [r for r in rows if r[key] is not None]
        if not valid:
            return None
        return min(valid, key=lambda r: r[key]) if lower_is_better else max(valid, key=lambda r: r[key])

    print()
    print("Best by TF arm:      ", best("tf_arm"))
    print("Best by TF hand:     ", best("tf_hand"))
    print("Best by AR arm:      ", best("ar_arm"))
    print("Best by AR hand:     ", best("ar_hand"))
    print("Best by vision_gain: ", best("vision_gain", lower_is_better=False))


if __name__ == "__main__":
    main()
