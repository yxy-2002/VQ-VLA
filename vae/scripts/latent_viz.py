"""
Visualize the 2D latent space of the hand-action VAE.

This script builds a small "latent atlas" around a trained checkpoint:

1. Dataset posterior means and sampled latent density
2. Decoder heatmaps over the 2D latent plane
3. Prototype windows (open / trigger / closing / closed) projected into z-space
4. Autoregressive rollout trajectories traced through latent space

Usage:
    python vae/scripts/latent_viz.py \
        --ckpt outputs/dim2_repro/checkpoint-20000.pth \
        --train_dir success/train \
        --test_dir success/test \
        --output_dir visualizations/latent_space/dim2_repro_best
"""

import argparse
import importlib.util
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.patches import Ellipse

_vae_root = os.path.join(os.path.dirname(__file__), "..")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_vae_root, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HandActionVAE = _load("model/hand_vae.py", "hand_vae").HandActionVAE

DEFAULT_INIT = np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
JOINT_NAMES = ["thumb_rot", "thumb_bend", "index", "middle", "ring", "pinky"]
ZH_FONT_CANDIDATES = [
    "Hiragino Sans GB",
    "Songti SC",
    "Arial Unicode MS",
]


def get_args():
    parser = argparse.ArgumentParser(description="Visualize 2D latent space of the hand-action VAE")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sample_repeats", type=int, default=4, help="Samples per dataset window for posterior cloud")
    parser.add_argument("--grid_res", type=int, default=180, help="Resolution of decoder heatmaps")
    parser.add_argument("--num_rollouts", type=int, default=256, help="Number of default-seed AR rollouts")
    parser.add_argument("--rollout_steps", type=int, default=100, help="Autoregressive rollout length")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lang", type=str, default="zh", choices=["zh", "en"], help="Figure language")
    return parser.parse_args()


def configure_plot_style(lang):
    plt.rcParams["axes.unicode_minus"] = False
    if lang != "zh":
        return

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ZH_FONT_CANDIDATES:
        if name in installed:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            return


def text(lang, zh, en):
    return zh if lang == "zh" else en


def infer_model_args(state_dict, default_window_size=8):
    kwargs = {}
    kwargs["latent_dim"] = state_dict["fc_mu.weight"].shape[0]
    kwargs["hidden_dim"] = state_dict["fc_mu.weight"].shape[1]

    decoder_indices = sorted(
        int(m.group(1)) for k in state_dict if (m := re.match(r"decoder\.(\d+)\.weight$", k))
    )
    last_dec_idx = decoder_indices[-1]
    kwargs["action_dim"] = state_dict[f"decoder.{last_dec_idx}.weight"].shape[0]
    kwargs["num_hidden_layers"] = max(0, len(decoder_indices) - 2)

    if "encoder.net.0.conv.weight" in state_dict:
        kwargs["encoder_type"] = "causal_conv"
        kwargs["window_size"] = default_window_size
    else:
        kwargs["encoder_type"] = "mlp"
        in_features = state_dict["encoder.0.weight"].shape[1]
        kwargs["window_size"] = in_features // kwargs["action_dim"]

    return kwargs


def finger_mean(actions):
    return actions[..., 2:6].mean(axis=-1)


def phase_label(current_finger, target_finger):
    if current_finger < 0.05 and target_finger < 0.05:
        return 0  # open hold
    if current_finger < 0.05 <= target_finger:
        return 1  # trigger
    if target_finger < 0.5:
        return 2  # closing
    return 3      # closed


def build_windows(data_dir, split_name, window_size):
    data_dir = Path(data_dir)
    traj_files = sorted(data_dir.glob("trajectory_*_demo_expert.pt"))

    rows = []
    for path in traj_files:
        traj_id = int(path.name.split("trajectory_")[1].split("_")[0])
        data = torch.load(path, map_location="cpu", weights_only=False)
        actions = data["actions"][:, 0, 6:12].float().numpy()
        total_steps = actions.shape[0]

        for t in range(total_steps):
            start = t - window_size + 1
            if start < 0:
                pad = np.repeat(actions[0:1], -start, axis=0)
                window = np.concatenate([pad, actions[: t + 1]], axis=0)
            else:
                window = actions[start : t + 1]

            target = actions[t + 1] if t + 1 < total_steps else actions[t]
            current = window[-1]
            current_f = float(finger_mean(current))
            target_f = float(finger_mean(target))
            rows.append(
                {
                    "split": split_name,
                    "traj_id": traj_id,
                    "step": t,
                    "window": window.astype(np.float32),
                    "target": target.astype(np.float32),
                    "current_finger": current_f,
                    "target_finger": target_f,
                    "phase": phase_label(current_f, target_f),
                }
            )

    return rows


@torch.no_grad()
def encode_windows(model, windows, device, batch_size=512):
    mu_chunks = []
    log_var_chunks = []
    for start in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[start : start + batch_size]).to(device)
        mu, log_var = model.encode(batch)
        mu_chunks.append(mu.cpu().numpy())
        log_var_chunks.append(log_var.cpu().numpy())
    return np.concatenate(mu_chunks, axis=0), np.concatenate(log_var_chunks, axis=0)


def posterior_samples(mu, log_var, repeats, rng):
    mu_rep = np.repeat(mu, repeats, axis=0)
    std_rep = np.repeat(np.exp(0.5 * log_var), repeats, axis=0)
    eps = rng.standard_normal(size=mu_rep.shape).astype(np.float32)
    return mu_rep + std_rep * eps


@torch.no_grad()
def decode_grid(model, xlim, ylim, grid_res, device):
    xs = np.linspace(xlim[0], xlim[1], grid_res, dtype=np.float32)
    ys = np.linspace(ylim[0], ylim[1], grid_res, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    z = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)

    z_tensor = torch.from_numpy(z).to(device)
    decoded = []
    for start in range(0, len(z), 4096):
        batch = z_tensor[start : start + 4096]
        decoded.append(model.decode(batch).cpu().numpy())
    decoded = np.concatenate(decoded, axis=0)

    return xx, yy, decoded.reshape(grid_res, grid_res, -1)


def compute_latent_bounds(mu, z_samples):
    all_points = np.concatenate([mu, z_samples], axis=0)
    x_lo, x_hi = np.quantile(all_points[:, 0], [0.005, 0.995])
    y_lo, y_hi = np.quantile(all_points[:, 1], [0.005, 0.995])
    x_pad = 0.12 * (x_hi - x_lo + 1e-6)
    y_pad = 0.12 * (y_hi - y_lo + 1e-6)
    return (float(x_lo - x_pad), float(x_hi + x_pad)), (float(y_lo - y_pad), float(y_hi + y_pad))


def add_prior_guides(ax):
    for radius, alpha in [(1.0, 0.25), (2.0, 0.15)]:
        circle = plt.Circle((0.0, 0.0), radius, fill=False, color="black", linestyle="--", alpha=alpha)
        ax.add_patch(circle)
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.15)
    ax.axvline(0.0, color="black", linewidth=0.5, alpha=0.15)


def add_diag_gaussian(ax, mu, log_var, color, label=None):
    std = np.exp(0.5 * log_var)
    ellipse = Ellipse(
        xy=mu,
        width=4.0 * std[0],
        height=4.0 * std[1],
        angle=0.0,
        facecolor=color,
        edgecolor=color,
        alpha=0.18,
        linewidth=2.0,
        label=label,
    )
    ax.add_patch(ellipse)
    ax.scatter(mu[0], mu[1], s=36, c=[color], edgecolors="black", linewidths=0.6, zorder=4)


def choose_prototypes(rows):
    open_candidates = [r for r in rows if r["current_finger"] < 0.02 and r["target_finger"] < 0.02]
    trigger_candidates = [r for r in rows if r["current_finger"] < 0.02 and r["target_finger"] > 0.08]
    closing_candidates = [r for r in rows if 0.15 < r["current_finger"] < 0.45 and r["target_finger"] > r["current_finger"]]
    closed_candidates = [r for r in rows if r["current_finger"] > 0.55 and r["target_finger"] > 0.55]

    def choose_middle(candidates, key_fn):
        candidates = sorted(candidates, key=key_fn)
        return candidates[len(candidates) // 2]

    return [
        ("Open hold", choose_middle(open_candidates, lambda r: (r["target_finger"], r["step"]))),
        ("Trigger", choose_middle(trigger_candidates, lambda r: (r["target_finger"], r["step"]))),
        ("Closing", choose_middle(closing_candidates, lambda r: (r["current_finger"], r["step"]))),
        ("Closed", choose_middle(closed_candidates, lambda r: (r["target_finger"], r["step"]))),
    ]


@torch.no_grad()
def prototype_posteriors(model, prototypes, window_size, device, rng, samples_per_proto=1200):
    out = []
    for title, row in prototypes:
        window = torch.from_numpy(row["window"]).unsqueeze(0).to(device)
        mu, log_var = model.encode(window)
        mu = mu.cpu().numpy()[0]
        log_var = log_var.cpu().numpy()[0]
        std = np.exp(0.5 * log_var)
        z = mu[None, :] + rng.standard_normal(size=(samples_per_proto, 2)).astype(np.float32) * std[None, :]
        z_tensor = torch.from_numpy(z).to(device)
        decoded = model.decode(z_tensor).cpu().numpy()
        out.append(
            {
                "title": title,
                "row": row,
                "mu": mu,
                "log_var": log_var,
                "samples": z,
                "decoded": decoded,
            }
        )
    return out


@torch.no_grad()
def rollout_default_seed(model, window_size, num_rollouts, rollout_steps, device, rng):
    init = torch.from_numpy(DEFAULT_INIT).to(device)
    actions = torch.zeros(num_rollouts, rollout_steps, 6, device=device)
    sampled_z = torch.zeros(num_rollouts, rollout_steps, 2, device=device)
    mu_hist = torch.zeros(num_rollouts, rollout_steps, 2, device=device)
    log_var_hist = torch.zeros(num_rollouts, rollout_steps, 2, device=device)

    actions[:, 0, :] = init

    for t in range(1, rollout_steps):
        window = actions[:, max(0, t - window_size) : t, :]
        if window.shape[1] < window_size:
            pad = actions[:, 0:1, :].expand(-1, window_size - window.shape[1], -1)
            window = torch.cat([pad, window], dim=1)

        mu, log_var = model.encode(window)
        std = torch.exp(0.5 * log_var)
        eps = torch.from_numpy(rng.standard_normal(size=std.shape).astype(np.float32)).to(device)
        z = mu + std * eps
        pred = model.decode(z)

        mu_hist[:, t, :] = mu
        log_var_hist[:, t, :] = log_var
        sampled_z[:, t, :] = z
        actions[:, t, :] = pred

    return {
        "actions": actions.cpu().numpy(),
        "sampled_z": sampled_z.cpu().numpy(),
        "mu": mu_hist.cpu().numpy(),
        "log_var": log_var_hist.cpu().numpy(),
    }


def select_representative_rollouts(rollout_actions):
    f = finger_mean(rollout_actions)
    close_mask = f > 0.2
    ever_close = close_mask.any(axis=1)
    first_close = np.where(ever_close, close_mask.argmax(axis=1), f.shape[1])

    selected = []
    success_indices = np.where(ever_close)[0]
    if len(success_indices) > 0:
        close_steps = first_close[success_indices]
        for label, quantile in [("Early close", 0.10), ("Median close", 0.50), ("Late close", 0.90)]:
            target = np.quantile(close_steps, quantile)
            idx = success_indices[np.argmin(np.abs(close_steps - target))]
            if idx not in [item["index"] for item in selected]:
                selected.append({"label": label, "index": int(idx)})

    failure_indices = np.where(~ever_close)[0]
    if len(failure_indices) > 0:
        selected.append({"label": "Still open at max step", "index": int(failure_indices[0])})
    elif len(success_indices) > 0:
        idx = success_indices[np.argmax(first_close[success_indices])]
        if idx not in [item["index"] for item in selected]:
            selected.append({"label": "Latest successful close", "index": int(idx)})

    return selected, ever_close, first_close


def save_summary_figure(path, lang, mu, current_f, target_f, phases, z_samples, xx, yy, decoded):
    grid_finger = finger_mean(decoded)
    grid_thumb_bend = decoded[..., 1]
    phase_names = ["open hold", "trigger", "closing", "closed"]
    phase_colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)

    panels = [
        (
            axes[0, 0],
            current_f,
            text(lang, "后验均值（mu），按当前四指均值着色", "Posterior means (mu), colored by current finger mean"),
        ),
        (
            axes[0, 1],
            target_f,
            text(lang, "后验均值（mu），按下一步四指均值着色", "Posterior means (mu), colored by next-step finger mean"),
        ),
    ]
    for ax, colors, title in panels:
        sc = ax.scatter(mu[:, 0], mu[:, 1], c=colors, cmap="viridis", s=9, alpha=0.65, linewidths=0)
        add_prior_guides(ax)
        ax.set_title(title)
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        fig.colorbar(sc, ax=ax, shrink=0.82, label=text(lang, "四指均值", "finger mean"))

    ax = axes[0, 2]
    hb = ax.hexbin(z_samples[:, 0], z_samples[:, 1], gridsize=55, cmap="magma", mincnt=1)
    add_prior_guides(ax)
    ax.set_title(text(lang, "从 q(z|window) 采样得到的隐变量密度", "Sampled latent density from q(z|window)"))
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    fig.colorbar(hb, ax=ax, shrink=0.82, label=text(lang, "采样次数", "sample count"))

    ax = axes[1, 0]
    im = ax.imshow(
        grid_finger,
        origin="lower",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap="viridis",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )
    contours = ax.contour(xx, yy, grid_finger, levels=[0.05, 0.2, 0.5], colors="white", linewidths=1.0)
    ax.clabel(contours, inline=True, fontsize=8, fmt={0.05: "0.05", 0.2: "0.2", 0.5: "0.5"})
    add_prior_guides(ax)
    ax.set_title(text(lang, "Decoder 地图：二维隐空间上的解码四指均值", "Decoder map: decoded finger mean over the latent plane"))
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    fig.colorbar(im, ax=ax, shrink=0.82, label=text(lang, "解码四指均值", "decoded finger mean"))

    ax = axes[1, 1]
    im = ax.imshow(
        grid_thumb_bend,
        origin="lower",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap="cividis",
        aspect="auto",
    )
    add_prior_guides(ax)
    ax.set_title(text(lang, "Decoder 地图：二维隐空间上的解码拇指弯曲", "Decoder map: decoded thumb bend over the latent plane"))
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    fig.colorbar(im, ax=ax, shrink=0.82, label=text(lang, "解码拇指弯曲", "decoded thumb bend"))

    ax = axes[1, 2]
    phase_names = (
        ["张开保持", "触发瞬间", "合拢过程中", "已合拢"]
        if lang == "zh"
        else ["open hold", "trigger", "closing", "closed"]
    )
    for phase_id, name, color in zip(range(4), phase_names, phase_colors):
        mask = phases == phase_id
        ax.scatter(mu[mask, 0], mu[mask, 1], s=10, alpha=0.72, c=color, label=name, linewidths=0)
    add_prior_guides(ax)
    ax.set_title(text(lang, "按运动阶段划分的后验均值", "Posterior means split by motion phase"))
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.legend(loc="best", fontsize=8)

    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_prototype_figure(path, lang, prototypes, xx, yy, decoded):
    grid_finger = finger_mean(decoded)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)

    for ax, proto in zip(axes.flat, prototypes):
        row = proto["row"]
        colors = finger_mean(proto["decoded"])
        ax.imshow(
            grid_finger,
            origin="lower",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap="viridis",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            alpha=0.88,
        )
        ax.scatter(proto["samples"][:, 0], proto["samples"][:, 1], c=colors, cmap="coolwarm", s=7, alpha=0.25, linewidths=0)
        add_diag_gaussian(ax, proto["mu"], proto["log_var"], color="#ffcc00")
        add_prior_guides(ax)
        ax.set_title(
            f"{proto['title']} | traj {row['traj_id']} step {row['step']}\n"
            + text(
                lang,
                f"当前四指均值={row['current_finger']:.3f}, 下一步四指均值={row['target_finger']:.3f}",
                f"current={row['current_finger']:.3f}, target={row['target_finger']:.3f}",
            )
        )
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")

    fig.suptitle(
        text(lang, "4 类原型窗口在二维隐空间中的后验分布", "Posterior prototypes projected onto the decoded finger-closure map"),
        fontsize=14,
    )
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_path(ax, lang, z_path, title, close_step):
    steps = np.arange(len(z_path))
    ax.plot(z_path[:, 0], z_path[:, 1], color="white", linewidth=1.4, alpha=0.65)
    sc = ax.scatter(z_path[:, 0], z_path[:, 1], c=steps, cmap="plasma", s=14, linewidths=0)
    ax.scatter(z_path[0, 0], z_path[0, 1], marker="*", s=90, c="cyan", edgecolors="black", linewidths=0.8, zorder=5)
    ax.scatter(z_path[-1, 0], z_path[-1, 1], marker="o", s=42, c="red", edgecolors="black", linewidths=0.8, zorder=5)
    close_text = (
        text(lang, f"合拢发生在 step={close_step}", f"close step={close_step}")
        if close_step < len(z_path)
        else text(lang, "直到最大步数仍未跨过 0.2", "never crossed 0.2")
    )
    ax.set_title(f"{title}\n{close_text}")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    return sc


def save_rollout_overview(path, lang, xx, yy, decoded, rollout, ever_close, first_close):
    grid_finger = finger_mean(decoded)
    sampled_z = rollout["sampled_z"]

    success_idx = np.where(ever_close)[0][:28]
    fail_idx = np.where(~ever_close)[0][:28]
    if len(fail_idx) == 0:
        fail_idx = np.where(ever_close)[0][-12:]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    for ax, indices, title in [
        (
            axes[0],
            success_idx,
            text(lang, "成功闭合的 rollout：采样隐空间路径", "Successful rollouts: sampled latent paths"),
        ),
        (
            axes[1],
            fail_idx,
            text(lang, "到末尾仍张开的 rollout：采样隐空间路径", "Open-tail rollouts: sampled latent paths"),
        ),
    ]:
        ax.imshow(
            grid_finger,
            origin="lower",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap="viridis",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            alpha=0.88,
        )
        for idx in indices:
            z = sampled_z[idx, 1:, :]
            ax.plot(z[:, 0], z[:, 1], color="white", linewidth=0.8, alpha=0.22)
        add_prior_guides(ax)
        ax.set_title(title)
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")

    ax = axes[2]
    bins = np.arange(0, rollout["actions"].shape[1] + 1, 4)
    ax.hist(first_close[ever_close], bins=bins, color="#4c78a8", alpha=0.9, label=text(lang, "合拢步数", "close step"))
    if np.any(~ever_close):
        ax.axvline(
            rollout["actions"].shape[1],
            color="#e45756",
            linestyle="--",
            linewidth=2,
            label=text(lang, "到末尾仍张开", "still open"),
        )
    ax.set_title(
        text(lang, "首次合拢步数分布\n（阈值：四指均值 > 0.2）", "Distribution of first close step\n(threshold: finger mean > 0.2)")
    )
    ax.set_xlabel(text(lang, "rollout 步数", "rollout step"))
    ax.set_ylabel(text(lang, "数量", "count"))
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)

    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_representative_rollouts(path, lang, xx, yy, decoded, rollout, selected, first_close):
    grid_finger = finger_mean(decoded)
    actions = rollout["actions"]
    sampled_z = rollout["sampled_z"]

    n_cols = len(selected)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 8.5), constrained_layout=True)
    if n_cols == 1:
        axes = np.array(axes).reshape(2, 1)

    last_sc = None
    for col, item in enumerate(selected):
        idx = item["index"]
        ax = axes[0, col]
        ax.imshow(
            grid_finger,
            origin="lower",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap="viridis",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            alpha=0.9,
        )
        z_path = sampled_z[idx, 1:, :]
        last_sc = plot_path(ax, lang, z_path, item["label"], int(first_close[idx]))
        add_prior_guides(ax)

        ax = axes[1, col]
        fm = finger_mean(actions[idx])
        ax.plot(fm, color="#4c78a8", linewidth=2.0, label=text(lang, "四指均值", "finger mean"))
        ax.axhline(0.2, color="#f58518", linestyle="--", linewidth=1.2, label=text(lang, "合拢阈值", "close threshold"))
        if first_close[idx] < len(fm):
            ax.axvline(first_close[idx], color="#e45756", linestyle=":", linewidth=1.4)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(text(lang, "rollout 步数", "rollout step"))
        ax.set_ylabel(text(lang, "四指均值", "finger mean"))
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.legend(loc="best", fontsize=8)

    if last_sc is not None:
        fig.colorbar(last_sc, ax=axes[0, :].tolist(), shrink=0.82, label=text(lang, "rollout 步数", "rollout step"))

    fig.suptitle(text(lang, "代表性自回归轨迹在隐空间中的路线", "Representative autoregressive rollouts through latent space"), fontsize=14)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_stats(path, model_kwargs, rows, mu, log_var, rollout, ever_close, first_close, selected):
    target_f = np.array([row["target_finger"] for row in rows], dtype=np.float32)
    current_f = np.array([row["current_finger"] for row in rows], dtype=np.float32)
    actions = rollout["actions"]
    final_f = finger_mean(actions)[:, -1]

    payload = {
        "model_kwargs": model_kwargs,
        "dataset": {
            "num_windows": len(rows),
            "current_finger_mean": float(current_f.mean()),
            "target_finger_mean": float(target_f.mean()),
            "mu_mean": mu.mean(axis=0).round(6).tolist(),
            "mu_std": mu.std(axis=0).round(6).tolist(),
            "log_var_mean": log_var.mean(axis=0).round(6).tolist(),
        },
        "rollouts": {
            "num_rollouts": int(actions.shape[0]),
            "rollout_steps": int(actions.shape[1]),
            "close_fraction_by_max_step": float(ever_close.mean()),
            "first_close_mean": float(first_close[ever_close].mean()) if np.any(ever_close) else None,
            "first_close_median": float(np.median(first_close[ever_close])) if np.any(ever_close) else None,
            "first_close_q90": float(np.quantile(first_close[ever_close], 0.9)) if np.any(ever_close) else None,
            "final_finger_mean_all": float(final_f.mean()),
            "final_finger_mean_closed": float(final_f[ever_close].mean()) if np.any(ever_close) else None,
            "representatives": [
                {"label": item["label"], "index": item["index"], "close_step": int(first_close[item["index"]])}
                for item in selected
            ],
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    args = get_args()
    configure_plot_style(args.lang)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_kwargs = infer_model_args(ckpt["model"])
    if model_kwargs["latent_dim"] != 2:
        raise ValueError(f"This visualization script expects latent_dim=2, got {model_kwargs['latent_dim']}")

    device = torch.device(args.device)
    model = HandActionVAE(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows = build_windows(args.train_dir, "train", model_kwargs["window_size"])
    rows += build_windows(args.test_dir, "test", model_kwargs["window_size"])

    windows = np.stack([row["window"] for row in rows], axis=0)
    current_f = np.array([row["current_finger"] for row in rows], dtype=np.float32)
    target_f = np.array([row["target_finger"] for row in rows], dtype=np.float32)
    phases = np.array([row["phase"] for row in rows], dtype=np.int64)

    mu, log_var = encode_windows(model, windows, device)
    z_samples = posterior_samples(mu, log_var, args.sample_repeats, rng)
    xlim, ylim = compute_latent_bounds(mu, z_samples)
    xx, yy, decoded = decode_grid(model, xlim, ylim, args.grid_res, device)

    prototypes = choose_prototypes(rows)
    proto_info = prototype_posteriors(model, prototypes, model_kwargs["window_size"], device, rng)

    rollout = rollout_default_seed(
        model,
        model_kwargs["window_size"],
        num_rollouts=args.num_rollouts,
        rollout_steps=args.rollout_steps,
        device=device,
        rng=rng,
    )
    selected, ever_close, first_close = select_representative_rollouts(rollout["actions"])

    summary_path = os.path.join(args.output_dir, "01_latent_summary.png")
    proto_path = os.path.join(args.output_dir, "02_latent_prototypes.png")
    overview_path = os.path.join(args.output_dir, "03_rollout_overview.png")
    reps_path = os.path.join(args.output_dir, "04_rollout_representatives.png")
    stats_path = os.path.join(args.output_dir, "latent_stats.json")

    if args.lang == "zh":
        for proto, zh_title in zip(
            proto_info,
            ["张开保持", "触发瞬间", "合拢过程", "已合拢"],
        ):
            proto["title"] = zh_title
        for item, zh_label in zip(
            selected,
            ["早闭合", "中位闭合", "晚闭合", "到最大步数仍张开"],
        ):
            item["label"] = zh_label

    save_summary_figure(summary_path, args.lang, mu, current_f, target_f, phases, z_samples, xx, yy, decoded)
    save_prototype_figure(proto_path, args.lang, proto_info, xx, yy, decoded)
    save_rollout_overview(overview_path, args.lang, xx, yy, decoded, rollout, ever_close, first_close)
    save_representative_rollouts(reps_path, args.lang, xx, yy, decoded, rollout, selected, first_close)
    write_stats(stats_path, model_kwargs, rows, mu, log_var, rollout, ever_close, first_close, selected)

    print(f"Saved: {summary_path}")
    print(f"Saved: {proto_path}")
    print(f"Saved: {overview_path}")
    print(f"Saved: {reps_path}")
    print(f"Saved: {stats_path}")


if __name__ == "__main__":
    main()
