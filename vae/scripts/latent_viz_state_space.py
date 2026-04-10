"""
Visualize the learned 2D latent space of the recurrent state-space hand prior.
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

DEFAULT_INIT = np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
ZH_FONT_CANDIDATES = ["Hiragino Sans GB", "Songti SC", "Arial Unicode MS"]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_vae_root, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_eval = _load("scripts/eval_state_space_chunk_prior.py", "eval_state_space_chunk_prior")
HandActionStateSpaceChunkPrior = _eval.HandActionStateSpaceChunkPrior
infer_model_args = _eval.infer_model_args
rollout_free_batch = _eval.rollout_free_batch
chunk_hazard_from_first_close = _eval.chunk_hazard_from_first_close


def get_args():
    parser = argparse.ArgumentParser(description="Visualize 2D latent space of the state-space hand prior")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--grid_res", type=int, default=180)
    parser.add_argument("--num_rollouts", type=int, default=512)
    parser.add_argument("--max_steps", type=int, default=100)
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


def finger_mean(actions):
    return actions[..., 2:6].mean(axis=-1)


def first_crossing(series, threshold):
    idx = np.where(series > threshold)[0]
    return int(idx[0]) if len(idx) > 0 else len(series)


def load_trajectories(test_dir):
    traj = []
    for path in sorted(Path(test_dir).glob("trajectory_*_demo_expert.pt")):
        actions = torch.load(path, map_location="cpu", weights_only=False)["actions"][:, 0, 6:12].float().numpy()
        traj.append({"path": path, "actions": actions})
    return traj


def build_target_chunk(actions, chunk_idx, horizon):
    start = chunk_idx * horizon + 1
    future = []
    for offset in range(horizon):
        idx = min(start + offset, len(actions) - 1)
        future.append(actions[idx])
    return np.stack(future, axis=0).astype(np.float32)


@torch.no_grad()
def collect_posterior_records(model, trajectories, threshold):
    device = next(model.parameters()).device
    records = []
    for traj_id, item in enumerate(trajectories):
        actions = item["actions"]
        seed_window = np.repeat(actions[0:1], model.window_size, axis=0).astype(np.float32)
        hidden = model.encode_history(torch.from_numpy(seed_window).unsqueeze(0).to(device))
        current_state = torch.from_numpy(actions[0]).unsqueeze(0).to(device)
        prev_z = torch.zeros(1, model.latent_dim, device=device)
        num_chunks = max(1, math.ceil((len(actions) - 1) / model.future_horizon))
        path_z = []

        for chunk_idx in range(num_chunks):
            target_chunk_np = build_target_chunk(actions, chunk_idx, model.future_horizon)
            target_chunk = torch.from_numpy(target_chunk_np).unsqueeze(0).to(device)
            cond = model.make_condition(hidden, current_state, prev_z)
            prior_mu, prior_log_var = model.prior(cond, prev_z)
            post_mu, post_log_var = model.posterior(target_chunk, cond, current_state, prev_z)
            recon = model.decode(post_mu, cond, current_state)

            chunk_f = finger_mean(target_chunk_np)
            current_f = float(finger_mean(current_state.cpu().numpy())[0])
            path_z.append(post_mu[0].cpu().numpy())
            records.append(
                {
                    "traj_id": traj_id,
                    "chunk_idx": chunk_idx + 1,
                    "post_mu": post_mu[0].cpu().numpy(),
                    "prior_mu": prior_mu[0].cpu().numpy(),
                    "current_finger": current_f,
                    "target_end_finger": float(chunk_f[-1]),
                    "target_first_close": first_crossing(chunk_f, threshold) + 1 if np.any(chunk_f > threshold) else model.future_horizon + 1,
                    "recon_end_finger": float(finger_mean(recon.cpu().numpy())[0, -1]),
                }
            )

            hidden = model.transition_hidden(hidden, target_chunk, current_state, post_mu)
            current_state = target_chunk[:, -1]
            prev_z = post_mu

        for rec in records[-num_chunks:]:
            rec["traj_path"] = np.stack(path_z, axis=0)
    return records


def latent_bounds(points):
    x_lo, x_hi = np.quantile(points[:, 0], [0.005, 0.995])
    y_lo, y_hi = np.quantile(points[:, 1], [0.005, 0.995])
    x_pad = 0.12 * (x_hi - x_lo + 1e-6)
    y_pad = 0.12 * (y_hi - y_lo + 1e-6)
    return (float(x_lo - x_pad), float(x_hi + x_pad)), (float(y_lo - y_pad), float(y_hi + y_pad))


@torch.no_grad()
def decode_reset_grid(model, xlim, ylim, grid_res):
    device = next(model.parameters()).device
    xs = np.linspace(xlim[0], xlim[1], grid_res, dtype=np.float32)
    ys = np.linspace(ylim[0], ylim[1], grid_res, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    z = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)

    seed = torch.from_numpy(DEFAULT_INIT).to(device).view(1, 1, -1).expand(1, model.window_size, -1)
    hidden = model.encode_history(seed)
    current_state = seed[:, -1]
    prev_z = torch.zeros(1, model.latent_dim, device=device)
    cond = model.make_condition(hidden, current_state, prev_z)
    cond = cond.expand(z.shape[0], -1)
    current_expand = current_state.expand(z.shape[0], -1)

    decoded = []
    batch_size = 4096
    for start in range(0, len(z), batch_size):
        z_batch = torch.from_numpy(z[start : start + batch_size]).to(device)
        cond_batch = cond[start : start + batch_size]
        cur_batch = current_expand[start : start + batch_size]
        pred = model.decode(z_batch, cond_batch, cur_batch)
        decoded.append(pred.cpu().numpy())
    decoded = np.concatenate(decoded, axis=0)
    decoded = decoded.reshape(grid_res, grid_res, model.future_horizon, model.action_dim)
    return xx, yy, decoded


def dataset_onset_and_profile(trajectories, threshold):
    onsets = []
    seqs = []
    for item in trajectories:
        fm = finger_mean(item["actions"])
        onsets.append(first_crossing(fm, threshold))
        seqs.append(fm)
    max_len = max(len(seq) for seq in seqs)
    padded = np.stack([np.pad(seq, (0, max_len - len(seq)), constant_values=seq[-1]) for seq in seqs], axis=0)
    return np.array(onsets), padded


def choose_representatives(first_close, ever):
    closed_idx = np.where(ever)[0]
    if len(closed_idx) == 0:
        return []
    order = closed_idx[np.argsort(first_close[closed_idx])]
    picks = []
    for label, q in [("较早合拢", 0.1), ("中位合拢", 0.5), ("较晚合拢", 0.9)]:
        idx = order[min(len(order) - 1, int(round((len(order) - 1) * q)))]
        picks.append((label, idx))
    return picks


def plot_dynamics_overview(path, lang, dataset_onsets, dataset_profile, rollout_actions, threshold):
    rollout_fm = finger_mean(rollout_actions)
    rollout_cross = rollout_fm > threshold
    rollout_ever = rollout_cross.any(axis=1)
    rollout_first = np.where(rollout_ever, rollout_cross.argmax(axis=1), rollout_fm.shape[1])
    rollout_chunk_hazard = chunk_hazard_from_first_close(rollout_first, 12, rollout_actions.shape[1])["hazards"][:4]
    dataset_chunk_hazard = chunk_hazard_from_first_close(dataset_onsets, 12, dataset_profile.shape[1])["hazards"][:4]
    compare_steps = dataset_profile.shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    bins = np.arange(0, max(compare_steps, rollout_actions.shape[1]) + 2, 2)
    axes[0, 0].hist(dataset_onsets, bins=bins, alpha=0.55, label=text(lang, "数据集", "dataset"), color="#6886c5")
    axes[0, 0].hist(rollout_first[rollout_ever], bins=bins, alpha=0.55, label=text(lang, "模型 rollout", "model rollout"), color="#dd8452")
    axes[0, 0].set_title(text(lang, "首次合拢时间分布", "First-close time distribution"))
    axes[0, 0].set_xlabel(text(lang, "步数", "step"))
    axes[0, 0].set_ylabel(text(lang, "数量", "count"))
    axes[0, 0].legend()

    xs = np.arange(1, 1 + len(dataset_chunk_hazard))
    axes[0, 1].plot(xs, dataset_chunk_hazard, marker="o", linewidth=2, label=text(lang, "数据集 hazard", "dataset hazard"), color="#6886c5")
    axes[0, 1].plot(xs, rollout_chunk_hazard, marker="o", linewidth=2, label=text(lang, "模型 hazard", "model hazard"), color="#dd8452")
    axes[0, 1].set_xticks(xs)
    axes[0, 1].set_ylim(0.0, 1.02)
    axes[0, 1].set_title(text(lang, "从 reset 开始的 chunk hazard", "Chunk hazard from reset"))
    axes[0, 1].set_xlabel(text(lang, "第几个 12 步 chunk", "12-step chunk index"))
    axes[0, 1].set_ylabel(text(lang, "条件合拢概率", "conditional close probability"))
    axes[0, 1].legend()

    steps = np.arange(compare_steps)
    axes[1, 0].plot(steps, np.quantile(dataset_profile, 0.1, axis=0), color="#91b8ff", label=text(lang, "数据 10% 分位", "data 10%"))
    axes[1, 0].plot(steps, np.quantile(dataset_profile, 0.5, axis=0), color="#315aa6", linewidth=2.3, label=text(lang, "数据中位数", "data median"))
    axes[1, 0].plot(steps, np.quantile(dataset_profile, 0.9, axis=0), color="#19335b", label=text(lang, "数据 90% 分位", "data 90%"))
    axes[1, 0].plot(steps, np.quantile(rollout_fm[:, :compare_steps], 0.1, axis=0), color="#f0b07a", label=text(lang, "模型 10% 分位", "model 10%"))
    axes[1, 0].plot(steps, np.quantile(rollout_fm[:, :compare_steps], 0.5, axis=0), color="#dd8452", linewidth=2.3, label=text(lang, "模型中位数", "model median"))
    axes[1, 0].plot(steps, np.quantile(rollout_fm[:, :compare_steps], 0.9, axis=0), color="#8c4c21", label=text(lang, "模型 90% 分位", "model 90%"))
    axes[1, 0].axhline(threshold, color="black", linestyle="--", alpha=0.2)
    axes[1, 0].set_title(text(lang, "四指均值时间曲线（对齐到数据长度）", "Finger-mean profiles aligned to dataset length"))
    axes[1, 0].set_xlabel(text(lang, "步数", "step"))
    axes[1, 0].set_ylabel(text(lang, "四指均值", "finger mean"))
    axes[1, 0].legend(ncol=2, fontsize=9)

    data_cum = np.array([(dataset_onsets <= t).mean() for t in range(compare_steps)])
    model_cum = np.array([(rollout_first <= t).mean() for t in range(compare_steps)])
    axes[1, 1].plot(steps, data_cum, color="#315aa6", linewidth=2.3, label=text(lang, "数据累计合拢率", "data cumulative close"))
    axes[1, 1].plot(steps, model_cum, color="#dd8452", linewidth=2.3, label=text(lang, "模型累计合拢率", "model cumulative close"))
    axes[1, 1].set_title(text(lang, "累计合拢比例", "Cumulative close ratio"))
    axes[1, 1].set_xlabel(text(lang, "步数", "step"))
    axes[1, 1].set_ylabel(text(lang, "比例", "ratio"))
    axes[1, 1].set_ylim(0.0, 1.02)
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.grid(alpha=0.15)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_latent_map(path, lang, records, rollout_latents, decoder_grid, xlim, ylim, threshold):
    post_mu = np.stack([r["post_mu"] for r in records], axis=0)
    chunk_idx = np.array([r["chunk_idx"] for r in records])
    end_finger = np.array([r["target_end_finger"] for r in records])
    first_close = np.array([r["target_first_close"] for r in records])
    xx, yy, decoded = decoder_grid
    end_map = finger_mean(decoded)[..., -1]
    first_map = np.apply_along_axis(
        lambda seq: first_crossing(seq, threshold) + 1 if np.any(seq > threshold) else decoded.shape[2] + 1,
        2,
        finger_mean(decoded),
    )
    rollout_chunk1 = rollout_latents[:, 0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    im = axes[0, 0].scatter(post_mu[:, 0], post_mu[:, 1], c=end_finger, cmap="viridis", s=18, alpha=0.75, linewidths=0)
    axes[0, 0].set_title(text(lang, "后验 latent：按 chunk 末端闭合幅度着色", "Posterior latent by chunk-end closure"))
    fig.colorbar(im, ax=axes[0, 0], label=text(lang, "末端四指均值", "end finger mean"))

    im = axes[0, 1].scatter(post_mu[:, 0], post_mu[:, 1], c=chunk_idx, cmap="plasma", s=18, alpha=0.75, linewidths=0)
    axes[0, 1].set_title(text(lang, "后验 latent：按轨迹中的 chunk 序号着色", "Posterior latent by chunk index"))
    fig.colorbar(im, ax=axes[0, 1], label=text(lang, "chunk 序号", "chunk index"))

    im = axes[1, 0].imshow(
        end_map,
        origin="lower",
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        aspect="auto",
        cmap="viridis",
    )
    axes[1, 0].scatter(rollout_chunk1[:, 0], rollout_chunk1[:, 1], s=8, c="white", alpha=0.24, linewidths=0)
    axes[1, 0].set_title(text(lang, "reset 条件解码地图：chunk 末端闭合幅度", "Reset decoder map: chunk-end closure"))
    fig.colorbar(im, ax=axes[1, 0], label=text(lang, "末端四指均值", "end finger mean"))

    im = axes[1, 1].imshow(
        first_map,
        origin="lower",
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        aspect="auto",
        cmap="cividis",
    )
    axes[1, 1].scatter(rollout_chunk1[:, 0], rollout_chunk1[:, 1], s=8, c="white", alpha=0.24, linewidths=0)
    axes[1, 1].set_title(text(lang, "reset 条件解码地图：chunk 内首次合拢步", "Reset decoder map: first close step"))
    fig.colorbar(im, ax=axes[1, 1], label=text(lang, "首次合拢步", "first close step"))

    for ax in axes.flat:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.grid(alpha=0.15)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_paths(path, lang, rollout_actions, rollout_latents, threshold):
    fm = finger_mean(rollout_actions)
    cross = fm > threshold
    ever = cross.any(axis=1)
    first = np.where(ever, cross.argmax(axis=1), fm.shape[1])
    reps = choose_representatives(first, ever)
    if not reps:
        return

    fig, axes = plt.subplots(2, len(reps), figsize=(5.2 * len(reps), 8.5), constrained_layout=True)
    if len(reps) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, (label, idx) in enumerate(reps):
        z_path = rollout_latents[idx]
        ax = axes[0, col]
        colors = np.linspace(0.0, 1.0, len(z_path))
        ax.scatter(z_path[:, 0], z_path[:, 1], c=colors, cmap="viridis", s=48, linewidths=0)
        ax.plot(z_path[:, 0], z_path[:, 1], color="white", linewidth=1.1, alpha=0.45)
        ax.set_title(f"{label}\n{text(lang, '首次合拢', 'first close')} = {int(first[idx])}")
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.grid(alpha=0.15)

        ax = axes[1, col]
        ax.plot(fm[idx], color="#dd8452", linewidth=2.2, label=text(lang, "四指均值", "finger mean"))
        ax.axhline(threshold, color="black", linestyle="--", alpha=0.2)
        ax.set_xlabel(text(lang, "步数", "step"))
        ax.set_ylabel(text(lang, "四指均值", "finger mean"))
        ax.grid(alpha=0.15)
        ax.legend()

    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = get_args()
    configure_plot_style(args.lang)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = HandActionStateSpaceChunkPrior(**infer_model_args(ckpt)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    trajectories = load_trajectories(args.test_dir)
    records = collect_posterior_records(model, trajectories, args.threshold)
    dataset_onsets, dataset_profile = dataset_onset_and_profile(trajectories, args.threshold)

    rollout = rollout_free_batch(model, args.num_rollouts, args.max_steps, deterministic=False)
    rollout_actions = rollout["actions"]
    rollout_latents = rollout["chunk_latents"]

    post_pts = np.stack([r["post_mu"] for r in records], axis=0)
    prior_pts = rollout_latents.reshape(-1, rollout_latents.shape[-1])
    all_pts = np.concatenate([post_pts, prior_pts], axis=0)
    xlim, ylim = latent_bounds(all_pts)
    decoder_grid = decode_reset_grid(model, xlim, ylim, args.grid_res)

    dynamics_path = out_dir / "01_起点动力学总览.png"
    latent_path = out_dir / "02_二维隐空间地图.png"
    rollout_path = out_dir / "03_代表性AR隐轨迹.png"

    plot_dynamics_overview(dynamics_path, args.lang, dataset_onsets, dataset_profile, rollout_actions, args.threshold)
    plot_latent_map(latent_path, args.lang, records, rollout_latents, decoder_grid, xlim, ylim, args.threshold)
    plot_rollout_paths(rollout_path, args.lang, rollout_actions, rollout_latents, args.threshold)

    rollout_fm = finger_mean(rollout_actions)
    rollout_cross = rollout_fm > args.threshold
    rollout_first = np.where(rollout_cross.any(axis=1), rollout_cross.argmax(axis=1), rollout_fm.shape[1])
    summary = {
        "checkpoint": args.ckpt,
        "files": [str(dynamics_path), str(latent_path), str(rollout_path)],
        "rollout_median": float(np.median(rollout_first)),
        "rollout_q90": float(np.quantile(rollout_first, 0.9)),
        "dataset_median": float(np.median(dataset_onsets)),
        "dataset_q90": float(np.quantile(dataset_onsets, 0.9)),
        "rollout_t48_mean": float(rollout_fm[:, min(dataset_profile.shape[1] - 1, 47)].mean()),
        "dataset_t48_mean": float(dataset_profile[:, min(dataset_profile.shape[1] - 1, 47)].mean()),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
