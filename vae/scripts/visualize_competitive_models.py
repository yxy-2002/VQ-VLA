"""
Compare several competitive hand-prior models with:

1) 5 GT trajectories and 50 autoregressive rollouts from each trajectory's initial
   action chunk, plotted over the 6 hand joints.
2) A 2D latent map that highlights where "closed-hand" chunks live.
"""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager

_vae_root = os.path.join(os.path.dirname(__file__), "..")
_proj_root = os.path.join(_vae_root, "..")

ZH_FONT_CANDIDATES = ["Hiragino Sans GB", "Songti SC", "Arial Unicode MS"]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_vae_root, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_flow_eval = _load("scripts/eval_chunk_flow.py", "eval_chunk_flow")
_ss_eval = _load("scripts/eval_state_space_chunk_prior.py", "eval_state_space_chunk_prior")
_chunk_eval = _load("scripts/eval_chunk.py", "eval_chunk")
_chunk_cvae_eval = _load("scripts/eval_chunk_cvae.py", "eval_chunk_cvae")
_step_eval = _load("scripts/eval.py", "eval_step")

HandActionVAE = _step_eval.HandActionVAE
HandActionChunkVAE = _chunk_eval.HandActionChunkVAE
HandActionChunkCVAE = _chunk_cvae_eval.HandActionChunkCVAE
HandActionChunkFlow = _flow_eval.HandActionChunkFlow
HandActionStateSpaceChunkPrior = _ss_eval.HandActionStateSpaceChunkPrior


DEFAULT_MODELS = [
    {
        "key": "vae_step",
        "label": "一步 VAE",
        "type": "vae_step",
        "ckpt": os.path.join(_proj_root, "outputs/dim2_noise002/checkpoint.pth"),
    },
    {
        "key": "chunk_h12_b1e3",
        "label": "Chunk-VAE",
        "type": "chunk_vae",
        "ckpt": os.path.join(_proj_root, "outputs/chunk_sweep/h12_b1e3/checkpoint.pth"),
    },
    {
        "key": "flow_post_best",
        "label": "Flow + posterior",
        "type": "flow",
        "ckpt": os.path.join(_proj_root, "outputs/flow_rollout_posterior_sweep/h12_r2e2_post_c025_t002/checkpoint-2000.pth"),
        "integration_steps": 24,
    },
    {
        "key": "state_space_rw_best",
        "label": "State-space（随机窗口）",
        "type": "state_space",
        "ckpt": os.path.join(_proj_root, "outputs/state_space_init/recon_f4/checkpoint.pth"),
    },
    {
        "key": "traj_hold",
        "label": "Trajectory-start",
        "type": "state_space",
        "ckpt": os.path.join(_proj_root, "outputs/state_space_traj/traj_hold/checkpoint.pth"),
    },
]


def get_args():
    parser = argparse.ArgumentParser(description="Visualize competitive hand-prior models")
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--models_json", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--closed_threshold", type=float, default=0.55)
    parser.add_argument("--num_traj", type=int, default=5)
    parser.add_argument("--num_rollouts", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--lang", type=str, default="zh", choices=["zh", "en"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model_specs(models_json):
    if models_json is None:
        return DEFAULT_MODELS
    with open(models_json, "r", encoding="utf-8") as f:
        models = json.load(f)
    if not isinstance(models, list):
        raise ValueError("models_json must contain a JSON list of model specs.")
    return models


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


def finger_mean(actions):
    return actions[..., 2:6].mean(axis=-1)


def first_crossing(series, threshold):
    idx = np.where(series > threshold)[0]
    return int(idx[0]) if len(idx) > 0 else len(series)


def load_trajectories(test_dir, threshold):
    rows = []
    for path in sorted(Path(test_dir).glob("trajectory_*_demo_expert.pt")):
        traj_id = int(path.name.split("trajectory_")[1].split("_")[0])
        actions = torch.load(path, map_location="cpu", weights_only=False)["actions"][:, 0, 6:12].float().numpy()
        fm = finger_mean(actions)
        onset = first_crossing(fm, threshold)
        rows.append(
            {
                "traj_id": traj_id,
                "path": str(path),
                "actions": actions,
                "length": int(actions.shape[0]),
                "onset": int(onset),
            }
        )
    return rows


def select_representative_trajectories(rows, num_traj):
    ordered = sorted(rows, key=lambda x: (x["onset"], x["traj_id"]))
    quantiles = np.linspace(0.0, 1.0, num_traj)
    picked = []
    used = set()
    for q in quantiles:
        center = min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))
        for delta in range(len(ordered)):
            for idx in [center - delta, center + delta]:
                if 0 <= idx < len(ordered) and ordered[idx]["traj_id"] not in used:
                    used.add(ordered[idx]["traj_id"])
                    picked.append(ordered[idx])
                    break
            else:
                continue
            break
    return picked


def build_seed_window(actions, window_size):
    take = min(window_size, actions.shape[0])
    if take == window_size:
        return actions[:window_size].astype(np.float32)
    pad = np.repeat(actions[:1], window_size - take, axis=0)
    return np.concatenate([pad, actions[:take]], axis=0).astype(np.float32)


def build_open_mean_seed(rows, window_size):
    starts = np.stack([row["actions"][0] for row in rows], axis=0).astype(np.float32)
    open_mean = starts.mean(axis=0)
    return np.repeat(open_mean[None, :], window_size, axis=0).astype(np.float32)


def padded_gt(actions, max_steps):
    if actions.shape[0] >= max_steps:
        return actions[:max_steps].astype(np.float32)
    pad = np.repeat(actions[-1:], max_steps - actions.shape[0], axis=0)
    return np.concatenate([actions, pad], axis=0).astype(np.float32)


def load_model(spec, device):
    ckpt = torch.load(spec["ckpt"], map_location="cpu", weights_only=False)
    if spec["type"] == "vae_step":
        model = HandActionVAE(**_step_eval.infer_model_args(ckpt["model"])).to(device)
    elif spec["type"] == "chunk_vae":
        model = HandActionChunkVAE(**_chunk_eval.infer_model_args(ckpt["model"])).to(device)
    elif spec["type"] == "chunk_cvae":
        args = ckpt.get("args", {})
        model = HandActionChunkCVAE(
            action_dim=args.get("action_dim", 6),
            window_size=args.get("window_size", 8),
            future_horizon=args.get("future_horizon", 12),
            hidden_dim=args.get("hidden_dim", 256),
            latent_dim=args.get("latent_dim", 2),
            beta=args.get("beta", 0.001),
            encoder_type=args.get("encoder_type", "mlp"),
            num_hidden_layers=args.get("num_hidden_layers", 1),
            free_bits=args.get("free_bits", 0.0),
        ).to(device)
    elif spec["type"] == "flow":
        model = HandActionChunkFlow(**_flow_eval.infer_model_args(ckpt)).to(device)
    elif spec["type"] == "state_space":
        model = HandActionStateSpaceChunkPrior(**_ss_eval.infer_model_args(ckpt)).to(device)
    else:
        raise ValueError(f"Unsupported model type: {spec['type']}")
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def rollout_from_seed(spec, model, seed_window, num_rollouts, max_steps):
    device = next(model.parameters()).device
    window_size = model.window_size
    actions = torch.zeros(num_rollouts, max_steps, model.action_dim, device=device)
    seed_tensor = torch.from_numpy(seed_window).to(device)
    actions[:, :window_size, :] = seed_tensor.unsqueeze(0)
    chunk_latents = []

    if spec["type"] == "vae_step":
        t = window_size
        while t < max_steps:
            window = actions[:, t - window_size : t, :]
            pred = model.predict(window, deterministic=False)
            actions[:, t, :] = pred
            t += 1
        return actions.cpu().numpy(), np.zeros((num_rollouts, 0, model.latent_dim), dtype=np.float32)

    if spec["type"] == "chunk_vae":
        t = window_size
        stride = int(spec.get("rollout_stride", model.future_horizon))
        stride = max(1, min(stride, model.future_horizon))
        while t < max_steps:
            window = actions[:, t - window_size : t, :]
            pred = model.predict(window, deterministic=False)
            take = min(stride, max_steps - t)
            actions[:, t : t + take, :] = pred[:, :take]
            t += take
        return actions.cpu().numpy(), np.zeros((num_rollouts, 0, 2), dtype=np.float32)

    if spec["type"] == "chunk_cvae":
        t = window_size
        stride = int(spec.get("rollout_stride", model.future_horizon))
        stride = max(1, min(stride, model.future_horizon))
        while t < max_steps:
            window = actions[:, t - window_size : t, :]
            pred = model.predict(window, deterministic=False)
            take = min(stride, max_steps - t)
            actions[:, t : t + take, :] = pred[:, :take]
            t += take
        return actions.cpu().numpy(), np.zeros((num_rollouts, 0, model.latent_dim), dtype=np.float32)

    if spec["type"] == "flow":
        t = window_size
        while t < max_steps:
            window = actions[:, t - window_size : t, :]
            pred, z = model.predict(
                window,
                num_integration_steps=spec.get("integration_steps", 24),
                return_latent=True,
            )
            take = min(model.future_horizon, max_steps - t)
            actions[:, t : t + take, :] = pred[:, :take]
            chunk_latents.append(z.cpu().numpy())
            t += take
    else:
        hidden = model.encode_history(seed_tensor.unsqueeze(0).expand(num_rollouts, -1, -1))
        current_state = actions[:, window_size - 1]
        prev_z = torch.zeros(num_rollouts, model.latent_dim, device=device)
        t = window_size
        while t < max_steps:
            pred, z, hidden, current_state = model.sample_step(
                hidden,
                current_state,
                prev_z,
                deterministic=False,
            )
            take = min(model.future_horizon, max_steps - t)
            actions[:, t : t + take, :] = pred[:, :take]
            chunk_latents.append(z.cpu().numpy())
            prev_z = z
            t += take

    latents = (
        np.stack(chunk_latents, axis=1)
        if chunk_latents
        else np.zeros((num_rollouts, 0, model.latent_dim), dtype=np.float32)
    )
    return actions.cpu().numpy(), latents


def non_overlapping_chunks(actions, horizon):
    num_chunks = max(1, math.ceil((len(actions) - 1) / horizon))
    out = []
    for chunk_idx in range(num_chunks):
        start = chunk_idx * horizon + 1
        future = []
        for offset in range(horizon):
            idx = min(start + offset, len(actions) - 1)
            future.append(actions[idx])
        out.append(np.stack(future, axis=0).astype(np.float32))
    return out


@torch.no_grad()
def collect_step_vae_latents(model, trajectories, threshold):
    device = next(model.parameters()).device
    records = []
    for row in trajectories:
        actions = row["actions"]
        for t in range(len(actions)):
            hist_end = t
            hist_start = hist_end - model.window_size + 1
            if hist_start < 0:
                pad = np.repeat(actions[:1], -hist_start, axis=0)
                history = np.concatenate([pad, actions[: hist_end + 1]], axis=0)
            else:
                history = actions[hist_start : hist_end + 1]
            history_t = torch.from_numpy(history).unsqueeze(0).to(device)
            mu, _ = model.encode(history_t)
            z = mu[0].cpu().numpy()
            target = actions[min(t + 1, len(actions) - 1)]
            end_f = float(finger_mean(target[None, :])[0])
            records.append(
                {
                    "traj_id": row["traj_id"],
                    "chunk_idx": t + 1,
                    "z": z,
                    "end_finger": end_f,
                    "is_closed": bool(end_f >= threshold),
                }
            )
    return records


@torch.no_grad()
def collect_flow_latents(model, trajectories, threshold):
    device = next(model.parameters()).device
    records = []
    for row in trajectories:
        actions = row["actions"]
        chunks = non_overlapping_chunks(actions, model.future_horizon)
        for chunk_idx, target in enumerate(chunks):
            end_t = chunk_idx * model.future_horizon
            hist_end = min(end_t, len(actions) - 1)
            hist_start = hist_end - model.window_size + 1
            if hist_start < 0:
                pad = np.repeat(actions[:1], -hist_start, axis=0)
                history = np.concatenate([pad, actions[: hist_end + 1]], axis=0)
            else:
                history = actions[hist_start : hist_end + 1]
            history_t = torch.from_numpy(history).unsqueeze(0).to(device)
            target_t = torch.from_numpy(target).unsqueeze(0).to(device)
            cond = model.encode_history(history_t)
            z = model.encode_future(target_t, cond=cond)[0].cpu().numpy()
            end_f = float(finger_mean(target)[-1])
            records.append(
                {
                    "traj_id": row["traj_id"],
                    "chunk_idx": chunk_idx + 1,
                    "z": z,
                    "end_finger": end_f,
                    "is_closed": bool(end_f >= threshold),
                }
            )
    return records


@torch.no_grad()
def collect_chunk_vae_latents(model, trajectories, threshold):
    device = next(model.parameters()).device
    records = []
    for row in trajectories:
        actions = row["actions"]
        chunks = non_overlapping_chunks(actions, model.future_horizon)
        for chunk_idx, target in enumerate(chunks):
            end_t = chunk_idx * model.future_horizon
            hist_end = min(end_t, len(actions) - 1)
            hist_start = hist_end - model.window_size + 1
            if hist_start < 0:
                pad = np.repeat(actions[:1], -hist_start, axis=0)
                history = np.concatenate([pad, actions[: hist_end + 1]], axis=0)
            else:
                history = actions[hist_start : hist_end + 1]
            history_t = torch.from_numpy(history).unsqueeze(0).to(device)
            mu, _ = model.encode(history_t)
            z = mu[0].cpu().numpy()
            end_f = float(finger_mean(target)[-1])
            records.append(
                {
                    "traj_id": row["traj_id"],
                    "chunk_idx": chunk_idx + 1,
                    "z": z,
                    "end_finger": end_f,
                    "is_closed": bool(end_f >= threshold),
                }
            )
    return records


@torch.no_grad()
def collect_chunk_cvae_latents(model, trajectories, threshold):
    device = next(model.parameters()).device
    records = []
    for row in trajectories:
        actions = row["actions"]
        chunks = non_overlapping_chunks(actions, model.future_horizon)
        for chunk_idx, target in enumerate(chunks):
            end_t = chunk_idx * model.future_horizon
            hist_end = min(end_t, len(actions) - 1)
            hist_start = hist_end - model.window_size + 1
            if hist_start < 0:
                pad = np.repeat(actions[:1], -hist_start, axis=0)
                history = np.concatenate([pad, actions[: hist_end + 1]], axis=0)
            else:
                history = actions[hist_start : hist_end + 1]
            history_t = torch.from_numpy(history).unsqueeze(0).to(device)
            target_t = torch.from_numpy(target).unsqueeze(0).to(device)
            enc = model.posterior(history_t, target_t)
            z = enc["post_mu"][0].cpu().numpy()
            end_f = float(finger_mean(target)[-1])
            records.append(
                {
                    "traj_id": row["traj_id"],
                    "chunk_idx": chunk_idx + 1,
                    "z": z,
                    "end_finger": end_f,
                    "is_closed": bool(end_f >= threshold),
                }
            )
    return records


@torch.no_grad()
def collect_state_space_latents(model, trajectories, threshold):
    device = next(model.parameters()).device
    records = []
    for row in trajectories:
        actions = row["actions"]
        seed_window = np.repeat(actions[:1], model.window_size, axis=0).astype(np.float32)
        hidden = model.encode_history(torch.from_numpy(seed_window).unsqueeze(0).to(device))
        current_state = torch.from_numpy(actions[0]).unsqueeze(0).to(device)
        prev_z = torch.zeros(1, model.latent_dim, device=device)
        chunks = non_overlapping_chunks(actions, model.future_horizon)
        for chunk_idx, target in enumerate(chunks):
            target_t = torch.from_numpy(target).unsqueeze(0).to(device)
            cond = model.make_condition(hidden, current_state, prev_z)
            post_mu, _ = model.posterior(target_t, cond, current_state, prev_z)
            z = post_mu[0].cpu().numpy()
            end_f = float(finger_mean(target)[-1])
            records.append(
                {
                    "traj_id": row["traj_id"],
                    "chunk_idx": chunk_idx + 1,
                    "z": z,
                    "end_finger": end_f,
                    "is_closed": bool(end_f >= threshold),
                }
            )
            hidden = model.transition_hidden(hidden, target_t, current_state, post_mu)
            current_state = target_t[:, -1]
            prev_z = post_mu
    return records


def plot_joint_rollouts(path, lang, model_label, traj_row, gt_seq, pred_seq, seed_len, max_steps):
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
    joint_names = [text(lang, f"关节 {i+1}", f"Joint {i+1}") for i in range(6)]
    x = np.arange(max_steps)
    end_line = min(traj_row["length"], max_steps) - 1

    all_values = np.concatenate([gt_seq[None], pred_seq], axis=0)
    y_min = float(all_values.min()) - 0.03
    y_max = float(all_values.max()) + 0.03

    for joint_idx, ax in enumerate(axes.flat):
        ax.axvspan(0, seed_len - 1, color="#f3e5ab", alpha=0.35, label=text(lang, "初始 chunk", "initial chunk"))
        for sample_idx in range(pred_seq.shape[0]):
            ax.plot(x, pred_seq[sample_idx, :, joint_idx], color="#4c78a8", alpha=0.12, linewidth=1.0)
        ax.plot(x, gt_seq[:, joint_idx], color="black", linewidth=2.3, label=text(lang, "GT", "GT"))
        ax.axvline(end_line, color="#888888", linestyle="--", linewidth=1.0, alpha=0.6, label=text(lang, "GT 结束", "GT end"))
        ax.set_title(joint_names[joint_idx])
        ax.set_xlabel(text(lang, "步数", "step"))
        ax.set_ylabel(text(lang, "关节角", "joint angle"))
        ax.set_xlim(0, max_steps - 1)
        ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.16)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    seen = {}
    uniq_handles, uniq_labels = [], []
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen[label] = True
            uniq_handles.append(handle)
            uniq_labels.append(label)
    fig.legend(uniq_handles, uniq_labels, loc="upper center", ncol=3)
    fig.suptitle(
        text(
            lang,
            f"{model_label} | 轨迹 {traj_row['traj_id']} | onset={traj_row['onset']} | 50 次 AR 采样",
            f"{model_label} | traj {traj_row['traj_id']} | onset={traj_row['onset']} | 50 AR samples",
        ),
        fontsize=14,
    )
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_joint_bundle(path, lang, model_label, traj_rows, pred_seq, seed_len, max_steps):
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
    joint_names = [text(lang, f"关节 {i+1}", f"Joint {i+1}") for i in range(6)]
    x = np.arange(max_steps)
    gt_colors = ["#111827", "#1d4ed8", "#059669", "#b45309", "#b91c1c"]

    gt_vals = [row["actions"][:max_steps] for row in traj_rows]
    all_gt = np.concatenate(gt_vals, axis=0) if gt_vals else np.zeros((0, 6), dtype=np.float32)
    all_values = np.concatenate([all_gt, pred_seq.reshape(-1, pred_seq.shape[-1])], axis=0)
    y_min = float(all_values.min()) - 0.03
    y_max = float(all_values.max()) + 0.03

    for joint_idx, ax in enumerate(axes.flat):
        ax.axvspan(0, seed_len - 1, color="#f3e5ab", alpha=0.35, label=text(lang, "初始 chunk", "initial chunk"))
        for sample_idx in range(pred_seq.shape[0]):
            label = text(lang, "50 次 AR 采样", "50 AR samples") if joint_idx == 0 and sample_idx == 0 else None
            ax.plot(x, pred_seq[sample_idx, :, joint_idx], color="#4c78a8", alpha=0.12, linewidth=1.0, label=label)
        for gt_idx, row in enumerate(traj_rows):
            seq = row["actions"][:max_steps]
            gt_x = np.arange(seq.shape[0])
            label = text(lang, f"GT 轨迹 {row['traj_id']}", f"GT traj {row['traj_id']}")
            ax.plot(
                gt_x,
                seq[:, joint_idx],
                color=gt_colors[gt_idx % len(gt_colors)],
                linewidth=2.2,
                alpha=0.95,
                label=label if joint_idx == 0 else None,
            )
        ax.set_title(joint_names[joint_idx])
        ax.set_xlabel(text(lang, "步数", "step"))
        ax.set_ylabel(text(lang, "关节角", "joint angle"))
        ax.set_xlim(0, max_steps - 1)
        ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.16)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    picked = " / ".join(str(row["traj_id"]) for row in traj_rows)
    onsets = " / ".join(str(row["onset"]) for row in traj_rows)
    fig.suptitle(
        text(
            lang,
            f"{model_label} | 同图 5 条 GT + 50 次 AR（traj={picked}，onset={onsets}）",
            f"{model_label} | 5 GT + 50 AR in one figure (traj={picked}, onset={onsets})",
        ),
        fontsize=14,
    )
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_closed_latent_map(path, lang, model_label, records, closed_threshold):
    z = np.stack([r["z"] for r in records], axis=0)
    end_f = np.array([r["end_finger"] for r in records])
    closed = end_f >= closed_threshold

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    scatter_kw = dict(s=22, alpha=0.75, linewidths=0)

    im = axes[0].scatter(z[:, 0], z[:, 1], c=end_f, cmap="viridis", **scatter_kw)
    axes[0].set_title(text(lang, "所有 chunk latent：按末端闭合幅度着色", "All chunk latents by end closure"))
    axes[0].set_xlabel("z1")
    axes[0].set_ylabel("z2")
    fig.colorbar(im, ax=axes[0], label=text(lang, "chunk 末端四指均值", "chunk-end finger mean"))

    axes[1].scatter(z[:, 0], z[:, 1], color="#c9c9c9", s=18, alpha=0.45, linewidths=0, label=text(lang, "全部 chunk", "all chunks"))
    if np.any(closed):
        axes[1].scatter(
            z[closed, 0],
            z[closed, 1],
            color="#d1495b",
            s=28,
            alpha=0.88,
            linewidths=0,
            label=text(lang, f"合拢手 chunk（末端>{closed_threshold:.2f}）", f"closed chunks (end>{closed_threshold:.2f})"),
        )
        center = z[closed].mean(axis=0)
        axes[1].scatter(center[0], center[1], color="#7f1d2d", s=90, marker="X", label=text(lang, "合拢手中心", "closed-hand center"))
    axes[1].set_title(text(lang, "二维 latent 中“合拢手”所在区域", "Closed-hand region in 2D latent"))
    axes[1].set_xlabel("z1")
    axes[1].set_ylabel("z2")
    axes[1].legend(loc="best", fontsize=9)

    x_lo, x_hi = np.quantile(z[:, 0], [0.01, 0.99])
    y_lo, y_hi = np.quantile(z[:, 1], [0.01, 0.99])
    x_pad = 0.15 * (x_hi - x_lo + 1e-6)
    y_pad = 0.15 * (y_hi - y_lo + 1e-6)
    for ax in axes:
        ax.set_xlim(float(x_lo - x_pad), float(x_hi + x_pad))
        ax.set_ylim(float(y_lo - y_pad), float(y_hi + y_pad))
        ax.grid(alpha=0.16)

    fig.suptitle(model_label, fontsize=14)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = get_args()
    configure_plot_style(args.lang)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_specs = load_model_specs(args.models_json)

    all_rows = load_trajectories(args.test_dir, args.threshold)
    picked_rows = select_representative_trajectories(all_rows, args.num_traj)

    summary = {
        "selected_trajectories": [
            {
                "traj_id": row["traj_id"],
                "length": row["length"],
                "onset": row["onset"],
                "path": row["path"],
            }
            for row in picked_rows
        ],
        "models": [],
    }

    for model_idx, spec in enumerate(model_specs):
        model = load_model(spec, device=device)
        model_dir = out_dir / spec["key"]
        model_dir.mkdir(parents=True, exist_ok=True)

        reset_seed = build_open_mean_seed(all_rows, model.window_size)
        torch.manual_seed(args.seed + model_idx * 1000 + 777)
        reset_pred_seq, _ = rollout_from_seed(
            spec,
            model,
            seed_window=reset_seed,
            num_rollouts=args.num_rollouts,
            max_steps=args.max_steps,
        )
        bundle_file = model_dir / f"gt{args.num_traj}_ar{args.num_rollouts}_same_plot_max{args.max_steps}.png"
        plot_joint_bundle(
            bundle_file,
            args.lang,
            spec["label"],
            picked_rows,
            reset_pred_seq,
            seed_len=model.window_size,
            max_steps=args.max_steps,
        )

        for traj_row in picked_rows:
            seed_window = build_seed_window(traj_row["actions"], model.window_size)
            gt_seq = padded_gt(traj_row["actions"], args.max_steps)
            torch.manual_seed(args.seed + model_idx * 1000 + traj_row["traj_id"])
            pred_seq, _ = rollout_from_seed(
                spec,
                model,
                seed_window=seed_window,
                num_rollouts=args.num_rollouts,
                max_steps=args.max_steps,
            )
            plot_joint_rollouts(
                model_dir / f"traj_{traj_row['traj_id']}_ar50_max{args.max_steps}.png",
                args.lang,
                spec["label"],
                traj_row,
                gt_seq,
                pred_seq,
                seed_len=model.window_size,
                max_steps=args.max_steps,
            )

        if spec["type"] == "vae_step":
            latent_records = collect_step_vae_latents(model, all_rows, args.closed_threshold)
        elif spec["type"] == "chunk_vae":
            latent_records = collect_chunk_vae_latents(model, all_rows, args.closed_threshold)
        elif spec["type"] == "chunk_cvae":
            latent_records = collect_chunk_cvae_latents(model, all_rows, args.closed_threshold)
        elif spec["type"] == "flow":
            latent_records = collect_flow_latents(model, all_rows, args.closed_threshold)
        else:
            latent_records = collect_state_space_latents(model, all_rows, args.closed_threshold)
        plot_closed_latent_map(
            model_dir / "latent_closed_map.png",
            args.lang,
            spec["label"],
            latent_records,
            closed_threshold=args.closed_threshold,
        )

        summary["models"].append(
            {
                "key": spec["key"],
                "label": spec["label"],
                "type": spec["type"],
                "checkpoint": spec["ckpt"],
                "output_dir": str(model_dir),
                "bundle_curve_file": str(bundle_file),
                "curve_files": [
                    str(model_dir / f"traj_{traj_row['traj_id']}_ar50_max{args.max_steps}.png")
                    for traj_row in picked_rows
                ],
                "latent_file": str(model_dir / "latent_closed_map.png"),
            }
        )

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
