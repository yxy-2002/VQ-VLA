"""
Visualize the 2D latent space of the conditional chunk flow model.
"""

import argparse
import importlib.util
import json
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


HandActionChunkFlow = _load("model/hand_chunk_flow.py", "hand_chunk_flow").HandActionChunkFlow


def get_args():
    parser = argparse.ArgumentParser(description="Visualize 2D latent space of the chunk flow model")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--grid_res", type=int, default=180)
    parser.add_argument("--num_open_samples", type=int, default=4096)
    parser.add_argument("--num_rollouts", type=int, default=256)
    parser.add_argument("--rollout_steps", type=int, default=100)
    parser.add_argument("--integration_steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def configure_plot_style():
    plt.rcParams["axes.unicode_minus"] = False
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ZH_FONT_CANDIDATES:
        if name in installed:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            return


def infer_model_args(ckpt):
    args = ckpt.get("args")
    if args is None:
        raise ValueError("Checkpoint is missing saved args.")
    return {
        "action_dim": args["action_dim"],
        "window_size": args["window_size"],
        "future_horizon": args["future_horizon"],
        "hidden_dim": args["hidden_dim"],
        "latent_dim": args["latent_dim"],
        "num_hidden_layers": args["num_hidden_layers"],
        "time_embed_dim": args["time_embed_dim"],
        "recon_weight": args["recon_weight"],
        "flow_weight": args["flow_weight"],
        "latent_reg_weight": args["latent_reg_weight"],
    }


def finger_mean(actions):
    return actions[..., 2:6].mean(axis=-1)


def first_crossing(series, threshold):
    idx = np.where(series > threshold)[0]
    return int(idx[0]) if len(idx) > 0 else len(series)


def build_rows(data_dir, split_name, window_size, future_horizon, threshold):
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
            for offset in range(1, future_horizon + 1):
                idx = min(t + offset, total_steps - 1)
                future.append(actions[idx])
            target = np.stack(future, axis=0).astype(np.float32)
            chunk_f = finger_mean(target)
            rows.append(
                {
                    "split": split_name,
                    "traj_id": traj_id,
                    "step": t,
                    "window": window.astype(np.float32),
                    "target": target,
                    "current_finger": float(finger_mean(window[-1])),
                    "future_end_finger": float(chunk_f[-1]),
                    "future_first_close": first_crossing(chunk_f, threshold) + 1,
                    "is_open_window": bool(np.abs(window[:, 2:6]).max() < 0.02),
                }
            )
    return rows


@torch.no_grad()
def encode_rows(model, rows, device, batch_size=512):
    zs, conds = [], []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        windows = torch.from_numpy(np.stack([r["window"] for r in batch_rows])).to(device)
        targets = torch.from_numpy(np.stack([r["target"] for r in batch_rows])).to(device)
        cond = model.encode_history(windows)
        z = model.encode_future(targets, cond=cond)
        zs.append(z.cpu().numpy())
        conds.append(cond.cpu().numpy())
    return np.concatenate(zs, axis=0), np.concatenate(conds, axis=0)


def latent_bounds(*arrays):
    pts = np.concatenate(arrays, axis=0)
    x_lo, x_hi = np.quantile(pts[:, 0], [0.005, 0.995])
    y_lo, y_hi = np.quantile(pts[:, 1], [0.005, 0.995])
    x_pad = 0.12 * (x_hi - x_lo + 1e-6)
    y_pad = 0.12 * (y_hi - y_lo + 1e-6)
    return (float(x_lo - x_pad), float(x_hi + x_pad)), (float(y_lo - y_pad), float(y_hi + y_pad))


def make_open_window(model, batch_size, device):
    window = np.repeat(DEFAULT_INIT[None, None, :], batch_size * model.window_size, axis=0)
    window = window.reshape(batch_size, model.window_size, model.action_dim)
    return torch.from_numpy(window).to(device)


@torch.no_grad()
def sample_open_condition(model, num_samples, integration_steps, device):
    window = make_open_window(model, num_samples, device)
    pred, z = model.predict(window, num_integration_steps=integration_steps, return_latent=True)
    return z.cpu().numpy(), pred.cpu().numpy()


@torch.no_grad()
def decode_grid_for_open(model, xlim, ylim, grid_res, device):
    xs = np.linspace(xlim[0], xlim[1], grid_res, dtype=np.float32)
    ys = np.linspace(ylim[0], ylim[1], grid_res, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    z = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    z_tensor = torch.from_numpy(z).to(device)
    cond = model.encode_history(make_open_window(model, 1, device)).expand(len(z), -1)
    decoded = []
    for start in range(0, len(z), 4096):
        decoded.append(model.decode(z_tensor[start : start + 4096], cond[start : start + 4096]).cpu().numpy())
    decoded = np.concatenate(decoded, axis=0)
    return xx, yy, decoded.reshape(grid_res, grid_res, model.future_horizon, model.action_dim)


@torch.no_grad()
def rollout_default_seed(model, num_rollouts, rollout_steps, integration_steps, device):
    horizon = model.future_horizon
    actions = torch.zeros(num_rollouts, rollout_steps, model.action_dim, device=device)
    actions[:, 0, :] = torch.from_numpy(DEFAULT_INIT).to(device)
    latents = []

    t = 1
    while t < rollout_steps:
        window = actions[:, max(0, t - model.window_size) : t, :]
        if window.shape[1] < model.window_size:
            pad = actions[:, 0:1, :].expand(-1, model.window_size - window.shape[1], -1)
            window = torch.cat([pad, window], dim=1)
        pred, z = model.predict(window, num_integration_steps=integration_steps, return_latent=True)
        take = min(horizon, rollout_steps - t)
        actions[:, t : t + take, :] = pred[:, :take, :]
        latents.append(z.cpu().numpy())
        t += take

    if latents:
        latents = np.stack(latents, axis=1)
    else:
        latents = np.zeros((num_rollouts, 0, model.latent_dim), dtype=np.float32)
    return actions.cpu().numpy(), latents


def summarize_first_close(fm, threshold):
    close = fm > threshold
    ever = close.any(axis=1)
    first = np.where(ever, close.argmax(axis=1), fm.shape[1])
    return ever, first


def dataset_rollout_reference(test_dir, threshold):
    all_fm = []
    onsets = []
    for path in sorted(Path(test_dir).glob("trajectory_*_demo_expert.pt")):
        actions = torch.load(path, map_location="cpu", weights_only=False)["actions"][:, 0, 6:12].float().numpy()
        fm = finger_mean(actions)
        all_fm.append(fm)
        onsets.append(first_crossing(fm, threshold))
    return np.array(onsets), all_fm


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


def plot_latent_overview(path, rows, z_all, z_open, open_sample_z, open_sample_pred, xlim, ylim, threshold):
    future_end = np.array([r["future_end_finger"] for r in rows])
    future_first = np.array([r["future_first_close"] for r in rows])
    open_end = finger_mean(open_sample_pred)[:, -1]
    open_first = np.array([first_crossing(seq, threshold) + 1 for seq in finger_mean(open_sample_pred)])

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    scatter_kw = dict(s=8, alpha=0.35, linewidths=0)

    im = axes[0, 0].scatter(z_all[:, 0], z_all[:, 1], c=future_end, cmap="viridis", **scatter_kw)
    axes[0, 0].set_title("后验编码点：按 chunk 末端四指均值着色")
    axes[0, 0].set_xlabel("z1")
    axes[0, 0].set_ylabel("z2")
    fig.colorbar(im, ax=axes[0, 0], label="末端四指均值")

    im = axes[0, 1].scatter(
        z_all[:, 0],
        z_all[:, 1],
        c=np.clip(future_first, 1, np.max(future_first)),
        cmap="plasma",
        **scatter_kw,
    )
    axes[0, 1].set_title("后验编码点：按 chunk 内首次合拢步着色")
    axes[0, 1].set_xlabel("z1")
    axes[0, 1].set_ylabel("z2")
    fig.colorbar(im, ax=axes[0, 1], label="首次合拢步")

    axes[1, 0].scatter(z_open[:, 0], z_open[:, 1], c="#b0b0b0", s=8, alpha=0.28, linewidths=0, label="数据中的全张开窗口")
    im = axes[1, 0].scatter(open_sample_z[:, 0], open_sample_z[:, 1], c=open_end, cmap="viridis", s=10, alpha=0.55, linewidths=0)
    axes[1, 0].set_title("全张开条件下：flow 采样终点与数据开手窗口对比")
    axes[1, 0].set_xlabel("z1")
    axes[1, 0].set_ylabel("z2")
    axes[1, 0].legend(loc="upper right", fontsize=9)
    fig.colorbar(im, ax=axes[1, 0], label="末端四指均值")

    im = axes[1, 1].scatter(open_sample_z[:, 0], open_sample_z[:, 1], c=open_first, cmap="cividis", s=10, alpha=0.6, linewidths=0)
    axes[1, 1].set_title("全张开条件下：flow 采样终点按首次合拢步着色")
    axes[1, 1].set_xlabel("z1")
    axes[1, 1].set_ylabel("z2")
    fig.colorbar(im, ax=axes[1, 1], label="首次合拢步")

    for ax in axes.flat:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(alpha=0.15)

    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_open_decoder_map(path, xx, yy, decoded_grid, z_open, open_sample_z, threshold):
    end_map = finger_mean(decoded_grid)[..., -1]
    first_map = np.apply_along_axis(lambda seq: first_crossing(seq, threshold) + 1, 2, finger_mean(decoded_grid))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)

    im = axes[0].imshow(
        end_map,
        origin="lower",
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        aspect="auto",
        cmap="viridis",
    )
    axes[0].scatter(open_sample_z[:, 0], open_sample_z[:, 1], s=6, c="white", alpha=0.18, linewidths=0)
    axes[0].set_title("开手条件 decoder 地图：chunk 末端四指均值")
    axes[0].set_xlabel("z1")
    axes[0].set_ylabel("z2")
    fig.colorbar(im, ax=axes[0], label="末端四指均值")

    im = axes[1].imshow(
        first_map,
        origin="lower",
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        aspect="auto",
        cmap="cividis",
    )
    axes[1].scatter(open_sample_z[:, 0], open_sample_z[:, 1], s=6, c="white", alpha=0.18, linewidths=0)
    axes[1].set_title("开手条件 decoder 地图：chunk 内首次合拢步")
    axes[1].set_xlabel("z1")
    axes[1].set_ylabel("z2")
    fig.colorbar(im, ax=axes[1], label="首次合拢步")

    axes[2].hexbin(z_open[:, 0], z_open[:, 1], gridsize=36, cmap="Greys", mincnt=1)
    axes[2].scatter(open_sample_z[:, 0], open_sample_z[:, 1], s=8, c="#cc5500", alpha=0.22, linewidths=0, label="flow 采样")
    axes[2].set_title("数据开手窗口后验点 vs. flow 采样终点")
    axes[2].set_xlabel("z1")
    axes[2].set_ylabel("z2")
    axes[2].legend(loc="upper right", fontsize=9)

    for ax in axes:
        ax.grid(alpha=0.15)

    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_analysis(path, rollout_actions, rollout_latents, dataset_onsets, threshold):
    fm = finger_mean(rollout_actions)
    ever, first = summarize_first_close(fm, threshold)
    reps = choose_representatives(first, ever)

    fig = plt.figure(figsize=(17, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_mean = fig.add_subplot(gs[0, 1])
    ax_traj = fig.add_subplot(gs[1, 0])
    ax_lat = fig.add_subplot(gs[1, 1])

    bins = np.arange(0, max(len(fm[0]), int(dataset_onsets.max()) + 3), 2)
    ax_hist.hist(dataset_onsets, bins=bins, alpha=0.55, label="数据集", color="#6c8ebf")
    ax_hist.hist(first[ever], bins=bins, alpha=0.55, label="flow rollout", color="#d17b0f")
    ax_hist.set_title("自动合手时机分布")
    ax_hist.set_xlabel("首次合拢步")
    ax_hist.set_ylabel("数量")
    ax_hist.legend()

    ax_mean.plot(np.quantile(fm, 0.1, axis=0), label="rollout 10% 分位", color="#7aa6c2")
    ax_mean.plot(np.quantile(fm, 0.5, axis=0), label="rollout 中位数", color="#0d6b8a", linewidth=2.2)
    ax_mean.plot(np.quantile(fm, 0.9, axis=0), label="rollout 90% 分位", color="#d17b0f")
    ax_mean.axhline(threshold, color="black", linestyle="--", alpha=0.25)
    ax_mean.set_title("四指均值随时间的统计轨迹")
    ax_mean.set_xlabel("rollout 步数")
    ax_mean.set_ylabel("四指均值")
    ax_mean.legend()

    for label, idx in reps:
        ax_traj.plot(fm[idx], label=f"{label} | 第 {first[idx]} 步", linewidth=2)
    ax_traj.axhline(threshold, color="black", linestyle="--", alpha=0.25)
    ax_traj.set_title("代表性自动合手轨迹")
    ax_traj.set_xlabel("rollout 步数")
    ax_traj.set_ylabel("四指均值")
    if reps:
        ax_traj.legend()

    for label, idx in reps:
        path_z = rollout_latents[idx]
        colors = np.linspace(0.0, 1.0, len(path_z))
        ax_lat.scatter(path_z[:, 0], path_z[:, 1], c=colors, cmap="viridis", s=40, alpha=0.9, label=label)
        ax_lat.plot(path_z[:, 0], path_z[:, 1], alpha=0.45)
    ax_lat.set_title("代表性 rollout 的 chunk-latent 路径")
    ax_lat.set_xlabel("z1")
    ax_lat.set_ylabel("z2")
    if reps:
        ax_lat.legend()

    for ax in [ax_hist, ax_mean, ax_traj, ax_lat]:
        ax.grid(alpha=0.15)

    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_integration_sensitivity(path, metrics, dataset_ref, open_ref):
    steps = [m["integration_steps"] for m in metrics]
    rollout_median = [m["rollout"]["first_close_median"] for m in metrics]
    rollout_q90 = [m["rollout"]["first_close_q90"] for m in metrics]
    rollout_final = [m["rollout"]["final_finger_mean_all"] for m in metrics]
    open_close = [m["open_seed"]["close_fraction_in_chunk"] for m in metrics]
    open_end = [m["open_seed"]["end_finger_mean"] for m in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    axes[0, 0].plot(steps, rollout_median, marker="o", label="rollout 中位首次合拢")
    axes[0, 0].axhline(dataset_ref["onset_median"], color="#555555", linestyle="--", label="数据集参考")
    axes[0, 0].set_title("积分步数 vs rollout 首次合拢中位数")
    axes[0, 0].set_xlabel("积分步数")
    axes[0, 0].set_ylabel("首次合拢中位数")
    axes[0, 0].legend()

    axes[0, 1].plot(steps, rollout_q90, marker="o", color="#d17b0f", label="rollout 首次合拢 q90")
    axes[0, 1].axhline(dataset_ref["onset_q90"], color="#555555", linestyle="--", label="数据集参考")
    axes[0, 1].set_title("积分步数 vs rollout 首次合拢 q90")
    axes[0, 1].set_xlabel("积分步数")
    axes[0, 1].set_ylabel("首次合拢 q90")
    axes[0, 1].legend()

    axes[1, 0].plot(steps, open_close, marker="o", color="#0d6b8a", label="开手条件 chunk 内合拢概率")
    axes[1, 0].axhline(open_ref["close_fraction_in_chunk"], color="#555555", linestyle="--", label="数据参考")
    axes[1, 0].set_title("积分步数 vs 开手条件下 chunk 内合拢概率")
    axes[1, 0].set_xlabel("积分步数")
    axes[1, 0].set_ylabel("合拢概率")
    axes[1, 0].legend()

    axes[1, 1].plot(steps, rollout_final, marker="o", color="#7f8c2f", label="rollout 末端四指均值")
    axes[1, 1].plot(steps, open_end, marker="s", color="#aa3377", label="开手条件 chunk 末端均值")
    axes[1, 1].axhline(dataset_ref["final_finger_mean"], color="#7f8c2f", linestyle="--", alpha=0.5)
    axes[1, 1].axhline(open_ref["end_finger_mean"], color="#aa3377", linestyle="--", alpha=0.5)
    axes[1, 1].set_title("积分步数对末端闭合幅度的影响")
    axes[1, 1].set_xlabel("积分步数")
    axes[1, 1].set_ylabel("四指均值")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.grid(alpha=0.15)

    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    args = get_args()
    configure_plot_style()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = HandActionChunkFlow(**infer_model_args(ckpt)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    train_rows = build_rows(args.train_dir, "train", model.window_size, model.future_horizon, args.threshold)
    test_rows = build_rows(args.test_dir, "test", model.window_size, model.future_horizon, args.threshold)
    rows = train_rows + test_rows
    z_all, _ = encode_rows(model, rows, device)
    open_rows = [r for r in rows if r["is_open_window"]]
    z_open, _ = encode_rows(model, open_rows, device)

    open_sample_z, open_sample_pred = sample_open_condition(model, args.num_open_samples, args.integration_steps, device)
    xlim, ylim = latent_bounds(z_all, open_sample_z, z_open)
    xx, yy, decoded_grid = decode_grid_for_open(model, xlim, ylim, args.grid_res, device)

    rollout_actions, rollout_latents = rollout_default_seed(
        model,
        num_rollouts=args.num_rollouts,
        rollout_steps=args.rollout_steps,
        integration_steps=args.integration_steps,
        device=device,
    )
    dataset_onsets, _ = dataset_rollout_reference(args.test_dir, args.threshold)

    sensitivity_metrics = []
    sensitivity_steps = [12, 24, 28, 32, 36, 40, 48, 64]
    for steps in sensitivity_steps:
        z_s, pred_s = sample_open_condition(model, 2048, steps, device)
        fm_s = finger_mean(pred_s)
        open_close = fm_s > args.threshold
        open_first = np.where(open_close.any(axis=1), open_close.argmax(axis=1) + 1, model.future_horizon + 1)

        actions_s, _ = rollout_default_seed(model, 512, args.rollout_steps, steps, device)
        rollout_fm = finger_mean(actions_s)
        ever_s, first_s = summarize_first_close(rollout_fm, args.threshold)

        sensitivity_metrics.append(
            {
                "integration_steps": steps,
                "rollout": {
                    "first_close_median": float(np.median(first_s[ever_s])) if np.any(ever_s) else None,
                    "first_close_q90": float(np.quantile(first_s[ever_s], 0.9)) if np.any(ever_s) else None,
                    "final_finger_mean_all": float(rollout_fm[:, -1].mean()),
                },
                "open_seed": {
                    "close_fraction_in_chunk": float(open_close.any(axis=1).mean()),
                    "first_close_median_if_close": float(np.median(open_first[open_close.any(axis=1)]))
                    if np.any(open_close.any(axis=1))
                    else None,
                    "end_finger_mean": float(fm_s[:, -1].mean()),
                },
            }
        )

    dataset_ref = {
        "onset_median": float(np.median(dataset_onsets)),
        "onset_q90": float(np.quantile(dataset_onsets, 0.9)),
        "final_finger_mean": float(np.mean([traj[-1] for traj in dataset_rollout_reference(args.test_dir, args.threshold)[1]])),
    }
    open_ref = {
        "close_fraction_in_chunk": float(
            np.mean([r["future_end_finger"] > args.threshold or r["future_first_close"] <= model.future_horizon for r in open_rows])
        ),
        "end_finger_mean": float(np.mean([r["future_end_finger"] for r in open_rows])),
    }

    plot_latent_overview(
        out_dir / "01_二维隐空间总览.png",
        rows,
        z_all,
        z_open,
        open_sample_z,
        open_sample_pred,
        xlim,
        ylim,
        args.threshold,
    )
    plot_open_decoder_map(
        out_dir / "02_开手条件解码地图.png",
        xx,
        yy,
        decoded_grid,
        z_open,
        open_sample_z,
        args.threshold,
    )
    plot_rollout_analysis(
        out_dir / "03_AR自动合手分析.png",
        rollout_actions,
        rollout_latents,
        dataset_onsets,
        args.threshold,
    )
    plot_integration_sensitivity(
        out_dir / "04_积分步数敏感性.png",
        sensitivity_metrics,
        dataset_ref,
        open_ref,
    )

    summary = {
        "checkpoint": args.ckpt,
        "model": infer_model_args(ckpt),
        "integration_steps_for_main_plots": args.integration_steps,
        "num_rows": len(rows),
        "num_open_rows": len(open_rows),
        "files": [
            str(out_dir / "01_二维隐空间总览.png"),
            str(out_dir / "02_开手条件解码地图.png"),
            str(out_dir / "03_AR自动合手分析.png"),
            str(out_dir / "04_积分步数敏感性.png"),
        ],
        "sensitivity_metrics": sensitivity_metrics,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
