"""
Visualize the 2D latent space of the chunked hand-action VAE.
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

DEFAULT_INIT = np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
ZH_FONT_CANDIDATES = ["Hiragino Sans GB", "Songti SC", "Arial Unicode MS"]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_vae_root, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HandActionChunkVAE = _load("model/hand_chunk_vae.py", "hand_chunk_vae").HandActionChunkVAE


def get_args():
    parser = argparse.ArgumentParser(description="Visualize 2D latent space of the chunked VAE")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--sample_repeats", type=int, default=4)
    parser.add_argument("--grid_res", type=int, default=180)
    parser.add_argument("--num_rollouts", type=int, default=256)
    parser.add_argument("--rollout_steps", type=int, default=100)
    parser.add_argument("--lang", type=str, default="zh", choices=["zh", "en"])
    parser.add_argument("--seed", type=int, default=42)
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


def infer_model_args(state_dict):
    kwargs = {}
    kwargs["latent_dim"] = state_dict["fc_mu.weight"].shape[0]
    kwargs["hidden_dim"] = state_dict["fc_mu.weight"].shape[1]
    decoder_indices = sorted(
        int(m.group(1)) for k in state_dict if (m := re.match(r"decoder\.(\d+)\.weight$", k))
    )
    kwargs["num_hidden_layers"] = max(0, len(decoder_indices) - 2)
    decoder_out = state_dict[f"decoder.{decoder_indices[-1]}.weight"].shape[0]
    encoder_in = state_dict["encoder.0.weight"].shape[1]
    kwargs["encoder_type"] = "mlp"

    for action_dim in [6, 12, 16, 20]:
        if encoder_in % action_dim == 0 and decoder_out % action_dim == 0:
            kwargs["action_dim"] = action_dim
            kwargs["window_size"] = encoder_in // action_dim
            kwargs["future_horizon"] = decoder_out // action_dim
            return kwargs
    raise ValueError("Failed to infer model arguments")


def finger_mean(actions):
    return actions[..., 2:6].mean(axis=-1)


def first_crossing(series, threshold):
    idx = np.where(series > threshold)[0]
    return int(idx[0]) if len(idx) > 0 else len(series)


def phase_label(current_f, chunk_f, threshold):
    onset = first_crossing(chunk_f, threshold)
    if current_f < 0.05 and onset == len(chunk_f):
        return 0
    if current_f < 0.05 and onset < len(chunk_f):
        return 1
    if current_f < threshold and chunk_f[-1] > current_f:
        return 2
    return 3


def build_rows(data_dir, split_name, window_size, threshold):
    rows = []
    for path in sorted(Path(data_dir).glob("trajectory_*_demo_expert.pt")):
        traj_id = int(path.name.split("trajectory_")[1].split("_")[0])
        actions = torch.load(path, map_location="cpu", weights_only=False)["actions"][:, 0, 6:12].float().numpy()
        total_steps = actions.shape[0]
        for t in range(total_steps):
            start = t - window_size + 1
            if start < 0:
                pad = np.repeat(actions[0:1], -start, axis=0)
                window = np.concatenate([pad, actions[: t + 1]], axis=0)
            else:
                window = actions[start : t + 1]

            future = []
            for offset in range(1, window_size + 1):
                idx = min(t + offset, total_steps - 1)
                future.append(actions[idx])
            target = np.stack(future, axis=0).astype(np.float32)
            current_f = float(finger_mean(window[-1]))
            chunk_f = finger_mean(target)
            rows.append(
                {
                    "split": split_name,
                    "traj_id": traj_id,
                    "step": t,
                    "window": window.astype(np.float32),
                    "target": target,
                    "current_finger": current_f,
                    "future_end_finger": float(chunk_f[-1]),
                    "future_first_close": first_crossing(chunk_f, threshold),
                    "phase": phase_label(current_f, chunk_f, threshold),
                }
            )
    return rows


@torch.no_grad()
def encode_windows(model, windows, device, batch_size=512):
    mu_chunks, lv_chunks = [], []
    for start in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[start : start + batch_size]).to(device)
        mu, lv = model.encode(batch)
        mu_chunks.append(mu.cpu().numpy())
        lv_chunks.append(lv.cpu().numpy())
    return np.concatenate(mu_chunks, axis=0), np.concatenate(lv_chunks, axis=0)


def posterior_samples(mu, log_var, repeats, rng):
    mu_rep = np.repeat(mu, repeats, axis=0)
    std_rep = np.repeat(np.exp(0.5 * log_var), repeats, axis=0)
    eps = rng.standard_normal(size=mu_rep.shape).astype(np.float32)
    return mu_rep + std_rep * eps


def latent_bounds(mu, z_samples):
    pts = np.concatenate([mu, z_samples], axis=0)
    x_lo, x_hi = np.quantile(pts[:, 0], [0.005, 0.995])
    y_lo, y_hi = np.quantile(pts[:, 1], [0.005, 0.995])
    x_pad = 0.12 * (x_hi - x_lo + 1e-6)
    y_pad = 0.12 * (y_hi - y_lo + 1e-6)
    return (float(x_lo - x_pad), float(x_hi + x_pad)), (float(y_lo - y_pad), float(y_hi + y_pad))


@torch.no_grad()
def decode_grid(model, xlim, ylim, grid_res, device):
    xs = np.linspace(xlim[0], xlim[1], grid_res, dtype=np.float32)
    ys = np.linspace(ylim[0], ylim[1], grid_res, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    z = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    z_tensor = torch.from_numpy(z).to(device)
    out = []
    for start in range(0, len(z), 4096):
        out.append(model.decode(z_tensor[start : start + 4096]).cpu().numpy())
    decoded = np.concatenate(out, axis=0)
    return xx, yy, decoded.reshape(grid_res, grid_res, model.future_horizon, model.action_dim)


def add_prior_guides(ax):
    for radius, alpha in [(1.0, 0.25), (2.0, 0.15)]:
        ax.add_patch(plt.Circle((0.0, 0.0), radius, fill=False, color="black", linestyle="--", alpha=alpha))
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.15)
    ax.axvline(0.0, color="black", linewidth=0.5, alpha=0.15)


def add_diag_gaussian(ax, mu, log_var, color):
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
    )
    ax.add_patch(ellipse)
    ax.scatter(mu[0], mu[1], s=36, c=[color], edgecolors="black", linewidths=0.6, zorder=4)


def choose_prototypes(rows, threshold):
    open_hold = [r for r in rows if r["current_finger"] < 0.05 and r["future_first_close"] == len(r["target"])]
    close_soon = [r for r in rows if r["current_finger"] < 0.05 and r["future_first_close"] < len(r["target"]) // 2]
    close_late = [r for r in rows if r["current_finger"] < 0.05 and len(r["target"]) // 2 <= r["future_first_close"] < len(r["target"])]
    closed = [r for r in rows if r["current_finger"] > 0.55 and r["future_end_finger"] > 0.55]

    def middle(cands, key_fn):
        cands = sorted(cands, key=key_fn)
        return cands[len(cands) // 2]

    return [
        ("张开保持", middle(open_hold, lambda r: (r["step"], r["traj_id"]))),
        ("较早触发", middle(close_soon, lambda r: (r["future_first_close"], r["step"]))),
        ("较晚触发", middle(close_late, lambda r: (r["future_first_close"], r["step"]))),
        ("已合拢", middle(closed, lambda r: (r["future_end_finger"], r["step"]))),
    ]


@torch.no_grad()
def prototype_posteriors(model, prototypes, device, rng, samples_per_proto=1200):
    out = []
    for title, row in prototypes:
        window = torch.from_numpy(row["window"]).unsqueeze(0).to(device)
        mu, lv = model.encode(window)
        mu = mu.cpu().numpy()[0]
        lv = lv.cpu().numpy()[0]
        std = np.exp(0.5 * lv)
        z = mu[None, :] + rng.standard_normal(size=(samples_per_proto, 2)).astype(np.float32) * std[None, :]
        decoded = model.decode(torch.from_numpy(z).to(device)).cpu().numpy()
        out.append({"title": title, "row": row, "mu": mu, "log_var": lv, "samples": z, "decoded": decoded})
    return out


@torch.no_grad()
def rollout_default_seed(model, num_rollouts, rollout_steps, device, rng):
    horizon = model.future_horizon
    num_chunks = int(np.ceil((rollout_steps - 1) / horizon))

    actions = torch.zeros(num_rollouts, rollout_steps, model.action_dim, device=device)
    actions[:, 0, :] = torch.from_numpy(DEFAULT_INIT).to(device)
    sampled_z = torch.zeros(num_rollouts, num_chunks, model.latent_dim, device=device)
    chunk_steps = []

    t = 1
    chunk_idx = 0
    while t < rollout_steps:
        window = actions[:, max(0, t - model.window_size) : t, :]
        if window.shape[1] < model.window_size:
            pad = actions[:, 0:1, :].expand(-1, model.window_size - window.shape[1], -1)
            window = torch.cat([pad, window], dim=1)
        mu, lv = model.encode(window)
        std = torch.exp(0.5 * lv)
        eps = torch.from_numpy(rng.standard_normal(size=std.shape).astype(np.float32)).to(device)
        z = mu + std * eps
        pred = model.decode(z)
        take = min(horizon, rollout_steps - t)
        actions[:, t : t + take, :] = pred[:, :take, :]
        sampled_z[:, chunk_idx, :] = z
        chunk_steps.append(t)
        t += take
        chunk_idx += 1

    return {"actions": actions.cpu().numpy(), "sampled_z": sampled_z.cpu().numpy(), "chunk_steps": np.array(chunk_steps)}


def select_representative_rollouts(actions, threshold):
    fm = finger_mean(actions)
    cross = fm > threshold
    ever = cross.any(axis=1)
    first = np.where(ever, cross.argmax(axis=1), fm.shape[1])

    selected = []
    success = np.where(ever)[0]
    if len(success) > 0:
        close_steps = first[success]
        for label, q in [("早闭合", 0.1), ("中位闭合", 0.5), ("晚闭合", 0.9)]:
            target = np.quantile(close_steps, q)
            idx = success[np.argmin(np.abs(close_steps - target))]
            if idx not in [item["index"] for item in selected]:
                selected.append({"label": label, "index": int(idx)})
    failure = np.where(~ever)[0]
    if len(failure) > 0:
        selected.append({"label": "到最大步数仍张开", "index": int(failure[0])})
    return selected, ever, first


def save_summary(path, lang, threshold, horizon, mu, current_f, future_end_f, future_first_close, phases, z_samples, xx, yy, decoded):
    end_finger = finger_mean(decoded[:, :, -1, :])
    onset_map = np.apply_along_axis(lambda s: first_crossing(s, threshold), 2, finger_mean(decoded))

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)

    panels = [
        (axes[0, 0], current_f, text(lang, "后验均值（mu），按当前四指均值着色", "Posterior means colored by current finger mean")),
        (axes[0, 1], future_end_f, text(lang, "后验均值（mu），按未来 chunk 末端四指均值着色", "Posterior means colored by chunk-end finger mean")),
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
        end_finger,
        origin="lower",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap="viridis",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )
    contours = ax.contour(xx, yy, end_finger, levels=[0.2, 0.5], colors="white", linewidths=1.0)
    ax.clabel(contours, inline=True, fontsize=8)
    add_prior_guides(ax)
    ax.set_title(text(lang, "Decoder 地图：未来 chunk 末端四指均值", "Decoder map: chunk-end finger mean"))
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    fig.colorbar(im, ax=ax, shrink=0.82, label=text(lang, "chunk 末端四指均值", "chunk-end finger mean"))

    ax = axes[1, 1]
    im = ax.imshow(
        onset_map,
        origin="lower",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap="plasma",
        aspect="auto",
        vmin=0.0,
        vmax=float(horizon),
    )
    add_prior_guides(ax)
    ax.set_title(text(lang, "Decoder 地图：未来 chunk 内首次合拢步", "Decoder map: first close step inside chunk"))
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    fig.colorbar(im, ax=ax, shrink=0.82, label=text(lang, "首次合拢步（未合拢=8）", "first close step (or horizon)"))

    ax = axes[1, 2]
    phase_names = ["张开保持", "触发", "合拢过程", "已合拢"] if lang == "zh" else ["open hold", "trigger", "closing", "closed"]
    phase_colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    for phase_id, name, color in zip(range(4), phase_names, phase_colors):
        mask = phases == phase_id
        ax.scatter(mu[mask, 0], mu[mask, 1], s=10, alpha=0.72, c=color, label=name, linewidths=0)
    add_prior_guides(ax)
    ax.set_title(text(lang, "按未来 chunk 类型划分的后验均值", "Posterior means split by future-chunk type"))
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.legend(loc="best", fontsize=8)

    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_prototypes(path, lang, threshold, xx, yy, decoded, proto_info):
    end_finger = finger_mean(decoded[:, :, -1, :])
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    for ax, proto in zip(axes.flat, proto_info):
        row = proto["row"]
        colors = finger_mean(proto["decoded"][:, -1, :])
        ax.imshow(
            end_finger,
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
        close_text = (
            text(lang, f"未来 chunk 首次合拢步={row['future_first_close']}", f"first close={row['future_first_close']}")
            if row["future_first_close"] < len(row["target"])
            else text(lang, "未来 chunk 内不合拢", "no close inside chunk")
        )
        ax.set_title(
            f"{proto['title']} | traj {row['traj_id']} step {row['step']}\n"
            + text(
                lang,
                f"当前四指均值={row['current_finger']:.3f}, 末端四指均值={row['future_end_finger']:.3f}, {close_text}",
                f"current={row['current_finger']:.3f}, end={row['future_end_finger']:.3f}, {close_text}",
            )
        )
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")

    fig.suptitle(text(lang, "未来 chunk 原型在二维隐空间中的后验分布", "Prototype posteriors over the latent plane"), fontsize=14)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_rollout_overview(path, lang, threshold, xx, yy, decoded, rollout, ever_close, first_close):
    end_finger = finger_mean(decoded[:, :, -1, :])
    sampled_z = rollout["sampled_z"]
    success_idx = np.where(ever_close)[0][:28]
    fail_idx = np.where(~ever_close)[0][:28]
    if len(fail_idx) == 0:
        fail_idx = np.where(ever_close)[0][-12:]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    for ax, indices, title in [
        (axes[0], success_idx, text(lang, "成功闭合的 rollout：chunk latent 路径", "Successful rollouts: chunk-latent paths")),
        (axes[1], fail_idx, text(lang, "到末尾仍张开的 rollout：chunk latent 路径", "Open-tail rollouts: chunk-latent paths")),
    ]:
        ax.imshow(
            end_finger,
            origin="lower",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap="viridis",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            alpha=0.88,
        )
        for idx in indices:
            z = sampled_z[idx]
            ax.plot(z[:, 0], z[:, 1], color="white", linewidth=0.9, alpha=0.22)
        add_prior_guides(ax)
        ax.set_title(title)
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")

    ax = axes[2]
    bins = np.arange(0, rollout["actions"].shape[1] + 1, 4)
    ax.hist(first_close[ever_close], bins=bins, color="#4c78a8", alpha=0.9, label=text(lang, "合拢步数", "close step"))
    if np.any(~ever_close):
        ax.axvline(rollout["actions"].shape[1], color="#e45756", linestyle="--", linewidth=2, label=text(lang, "到末尾仍张开", "still open"))
    ax.set_title(text(lang, "首次合拢步数分布\n（阈值：四指均值 > 0.2）", "Distribution of first close step"))
    ax.set_xlabel(text(lang, "rollout 步数", "rollout step"))
    ax.set_ylabel(text(lang, "数量", "count"))
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)

    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_representatives(path, lang, threshold, xx, yy, decoded, rollout, selected, first_close):
    end_finger = finger_mean(decoded[:, :, -1, :])
    actions = rollout["actions"]
    sampled_z = rollout["sampled_z"]
    chunk_steps = rollout["chunk_steps"]
    n_cols = len(selected)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.6 * n_cols, 8.5), constrained_layout=True)
    if n_cols == 1:
        axes = np.array(axes).reshape(2, 1)

    last_sc = None
    for col, item in enumerate(selected):
        idx = item["index"]
        ax = axes[0, col]
        ax.imshow(
            end_finger,
            origin="lower",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap="viridis",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            alpha=0.9,
        )
        z = sampled_z[idx]
        steps = chunk_steps
        ax.plot(z[:, 0], z[:, 1], color="white", linewidth=1.5, alpha=0.65)
        last_sc = ax.scatter(z[:, 0], z[:, 1], c=steps, cmap="plasma", s=28, linewidths=0)
        ax.scatter(z[0, 0], z[0, 1], marker="*", s=90, c="cyan", edgecolors="black", linewidths=0.8, zorder=5)
        ax.scatter(z[-1, 0], z[-1, 1], marker="o", s=42, c="red", edgecolors="black", linewidths=0.8, zorder=5)
        add_prior_guides(ax)
        close_text = (
            text(lang, f"合拢发生在 step={int(first_close[idx])}", f"close step={int(first_close[idx])}")
            if first_close[idx] < actions.shape[1]
            else text(lang, "直到最大步数仍未跨过 0.2", "never crossed 0.2")
        )
        ax.set_title(f"{item['label']}\n{close_text}")
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")

        ax = axes[1, col]
        fm = finger_mean(actions[idx])
        ax.plot(fm, color="#4c78a8", linewidth=2.0, label=text(lang, "四指均值", "finger mean"))
        ax.axhline(threshold, color="#f58518", linestyle="--", linewidth=1.2, label=text(lang, "合拢阈值", "close threshold"))
        if first_close[idx] < len(fm):
            ax.axvline(first_close[idx], color="#e45756", linestyle=":", linewidth=1.4)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(text(lang, "rollout 步数", "rollout step"))
        ax.set_ylabel(text(lang, "四指均值", "finger mean"))
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.legend(loc="best", fontsize=8)

    if last_sc is not None:
        fig.colorbar(last_sc, ax=axes[0, :].tolist(), shrink=0.82, label=text(lang, "chunk 起始步", "chunk start step"))

    fig.suptitle(text(lang, "代表性 chunk-level latent 轨迹", "Representative chunk-level latent rollouts"), fontsize=14)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_stats(path, model_kwargs, rows, mu, log_var, rollout, ever_close, first_close, selected):
    payload = {
        "model_kwargs": model_kwargs,
        "dataset": {
            "num_windows": len(rows),
            "mu_mean": mu.mean(axis=0).round(6).tolist(),
            "mu_std": mu.std(axis=0).round(6).tolist(),
            "log_var_mean": log_var.mean(axis=0).round(6).tolist(),
            "current_finger_mean": float(np.mean([r["current_finger"] for r in rows])),
            "future_end_finger_mean": float(np.mean([r["future_end_finger"] for r in rows])),
        },
        "rollouts": {
            "num_rollouts": int(rollout["actions"].shape[0]),
            "rollout_steps": int(rollout["actions"].shape[1]),
            "close_fraction_by_max_step": float(ever_close.mean()),
            "first_close_median": float(np.median(first_close[ever_close])) if np.any(ever_close) else None,
            "first_close_q90": float(np.quantile(first_close[ever_close], 0.9)) if np.any(ever_close) else None,
            "final_finger_mean_all": float(finger_mean(rollout["actions"])[:, -1].mean()),
            "representatives": [
                {"label": item["label"], "index": int(item["index"]), "close_step": int(first_close[item["index"]])}
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
        raise ValueError("This script expects latent_dim=2")
    device = torch.device(args.device)
    model = HandActionChunkVAE(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows = build_rows(args.train_dir, "train", model_kwargs["window_size"], args.threshold)
    rows += build_rows(args.test_dir, "test", model_kwargs["window_size"], args.threshold)
    windows = np.stack([r["window"] for r in rows], axis=0)

    current_f = np.array([r["current_finger"] for r in rows], dtype=np.float32)
    future_end_f = np.array([r["future_end_finger"] for r in rows], dtype=np.float32)
    future_first_close = np.array([r["future_first_close"] for r in rows], dtype=np.float32)
    phases = np.array([r["phase"] for r in rows], dtype=np.int64)

    mu, log_var = encode_windows(model, windows, device)
    z_samples = posterior_samples(mu, log_var, args.sample_repeats, rng)
    xlim, ylim = latent_bounds(mu, z_samples)
    xx, yy, decoded = decode_grid(model, xlim, ylim, args.grid_res, device)

    prototypes = choose_prototypes(rows, args.threshold)
    proto_info = prototype_posteriors(model, prototypes, device, rng)

    rollout = rollout_default_seed(model, args.num_rollouts, args.rollout_steps, device, rng)
    selected, ever_close, first_close = select_representative_rollouts(rollout["actions"], args.threshold)

    save_summary(
        os.path.join(args.output_dir, "01_chunk_latent_summary.png"),
        args.lang,
        args.threshold,
        model_kwargs["future_horizon"],
        mu,
        current_f,
        future_end_f,
        future_first_close,
        phases,
        z_samples,
        xx,
        yy,
        decoded,
    )
    save_prototypes(
        os.path.join(args.output_dir, "02_chunk_latent_prototypes.png"),
        args.lang,
        args.threshold,
        xx,
        yy,
        decoded,
        proto_info,
    )
    save_rollout_overview(
        os.path.join(args.output_dir, "03_chunk_rollout_overview.png"),
        args.lang,
        args.threshold,
        xx,
        yy,
        decoded,
        rollout,
        ever_close,
        first_close,
    )
    save_representatives(
        os.path.join(args.output_dir, "04_chunk_rollout_representatives.png"),
        args.lang,
        args.threshold,
        xx,
        yy,
        decoded,
        rollout,
        selected,
        first_close,
    )
    write_stats(
        os.path.join(args.output_dir, "chunk_latent_stats.json"),
        model_kwargs,
        rows,
        mu,
        log_var,
        rollout,
        ever_close,
        first_close,
        selected,
    )

    print(f"Saved analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
