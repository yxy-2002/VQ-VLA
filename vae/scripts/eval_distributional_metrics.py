"""
Evaluate hand-prior models with distributional metrics that are closer to the
intended prior-learning objective:

1) For ambiguous open-like contexts, compare the conditional future
   distribution of "when the hand first leaves the open basin".
2) From the common reset/open seed, compare long-horizon auto-closing timing,
   reopen violations, and closed-tail calibration.

This deliberately avoids task-specific "anti-drop" scores. The metrics focus on
distribution matching and support consistency instead of hand-crafted fixes for
one observed artifact.
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
_proj_root = os.path.join(_vae_root, "..")

ZH_FONT_CANDIDATES = ["Hiragino Sans GB", "Songti SC", "Arial Unicode MS"]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_vae_root, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_eval_old = _load("scripts/eval.py", "eval_old")
_eval_chunk = _load("scripts/eval_chunk.py", "eval_chunk")
_eval_flow = _load("scripts/eval_chunk_flow.py", "eval_chunk_flow")
_eval_ss = _load("scripts/eval_state_space_chunk_prior.py", "eval_state_space_chunk_prior")

HandActionVAE = _eval_old.HandActionVAE
HandActionChunkVAE = _eval_chunk.HandActionChunkVAE
HandActionChunkFlow = _eval_flow.HandActionChunkFlow
HandActionStateSpaceChunkPrior = _eval_ss.HandActionStateSpaceChunkPrior


DEFAULT_MODELS = [
    {
        "key": "vae_step",
        "label": "一步 VAE",
        "type": "vae_step",
        "ckpt": os.path.join(_proj_root, "outputs/dim2_noise002/checkpoint.pth"),
        "family": "general",
    },
    {
        "key": "chunk_h12_b1e3",
        "label": "Chunk-VAE",
        "type": "chunk_vae",
        "ckpt": os.path.join(_proj_root, "outputs/chunk_sweep/h12_b1e3/checkpoint.pth"),
        "family": "general",
    },
    {
        "key": "flow_post_best",
        "label": "Flow + posterior",
        "type": "flow",
        "ckpt": os.path.join(_proj_root, "outputs/flow_rollout_posterior_sweep/h12_r2e2_post_c025_t002/checkpoint-2000.pth"),
        "integration_steps": 24,
        "family": "general",
    },
    {
        "key": "state_space_rw_best",
        "label": "State-space（随机窗口）",
        "type": "state_space",
        "ckpt": os.path.join(_proj_root, "outputs/state_space_init/recon_f4/checkpoint.pth"),
        "family": "biased",
    },
    {
        "key": "traj_hold",
        "label": "Trajectory-start",
        "type": "state_space",
        "ckpt": os.path.join(_proj_root, "outputs/state_space_traj/traj_hold/checkpoint.pth"),
        "family": "biased",
    },
    {
        "key": "plateau_turbo_1000",
        "label": "Plateau（m2）",
        "type": "state_space",
        "ckpt": os.path.join(_proj_root, "outputs/state_space_traj_plateau/plateau_turbo_1000/checkpoint.pth"),
        "family": "biased",
    },
]


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate distributional hand-prior metrics")
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_plot", type=str, default=None)
    parser.add_argument("--models_json", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--condition_horizon", type=int, default=12)
    parser.add_argument("--condition_rollouts", type=int, default=50)
    parser.add_argument("--reset_rollouts", type=int, default=512)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--open_eps", type=float, default=0.02)
    parser.add_argument("--rise_threshold", type=float, default=0.05)
    parser.add_argument("--reopen_high", type=float, default=0.35)
    parser.add_argument("--reopen_low", type=float, default=0.15)
    parser.add_argument("--closed_tail", type=int, default=10)
    parser.add_argument("--closed_move_threshold", type=float, default=0.2)
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


def load_trajectories(dir_path):
    rows = []
    for path in sorted(Path(dir_path).glob("trajectory_*_demo_expert.pt")):
        traj_id = int(path.name.split("trajectory_")[1].split("_")[0])
        actions = torch.load(path, map_location="cpu", weights_only=False)["actions"][:, 0, 6:12].float().numpy()
        rows.append({"traj_id": traj_id, "actions": actions, "length": int(actions.shape[0])})
    return rows


def build_history_window(actions, end_idx, window_size):
    start = end_idx - window_size + 1
    if start >= 0:
        return actions[start : end_idx + 1].astype(np.float32)
    pad = np.repeat(actions[:1], -start, axis=0)
    return np.concatenate([pad, actions[: end_idx + 1]], axis=0).astype(np.float32)


def build_future(actions, start_idx, horizon):
    total = actions.shape[0]
    future = [actions[min(start_idx + off, total - 1)] for off in range(1, horizon + 1)]
    return np.stack(future, axis=0).astype(np.float32)


def load_progress_reference(train_dir, closed_tail, move_threshold):
    starts = []
    closed_tails = []
    for row in load_trajectories(train_dir):
        actions = row["actions"]
        starts.append(actions[0])
        if float(np.linalg.norm(actions[-1] - actions[0])) >= move_threshold:
            tail_len = min(closed_tail, actions.shape[0])
            closed_tails.append(actions[-tail_len:].mean(axis=0))
    if not closed_tails:
        raise ValueError("No closed trajectories found in the training split.")
    open_mean = np.stack(starts, axis=0).mean(axis=0).astype(np.float32)
    closed_mean = np.stack(closed_tails, axis=0).mean(axis=0).astype(np.float32)
    direction = (closed_mean - open_mean).astype(np.float32)
    norm_sq = float(np.sum(direction ** 2))
    if norm_sq <= 1e-8:
        raise ValueError("Closed and open reference states are too similar.")
    return {
        "open_mean": open_mean,
        "closed_mean": closed_mean,
        "direction": direction,
        "norm_sq": norm_sq,
    }


def progress_score(actions, ref):
    arr = np.asarray(actions, dtype=np.float32)
    return ((arr - ref["open_mean"]) @ ref["direction"]) / ref["norm_sq"]


def first_crossing(scores, threshold):
    idx = np.where(scores > threshold)[0]
    return int(idx[0] + 1) if len(idx) > 0 else len(scores) + 1


def pmf_from_delays(delays, horizon):
    arr = np.asarray(delays, dtype=np.int64)
    out = np.zeros(horizon + 1, dtype=np.float64)
    for k in range(1, horizon + 1):
        out[k - 1] = np.mean(arr == k)
    out[horizon] = np.mean(arr == (horizon + 1))
    return out


def hazard_from_delays(delays, horizon):
    arr = np.asarray(delays, dtype=np.int64)
    hazards = []
    active = np.ones(arr.shape[0], dtype=bool)
    for k in range(1, horizon + 1):
        denom = int(active.sum())
        if denom == 0:
            hazards.append(None)
        else:
            event = active & (arr == k)
            hazards.append(float(event.sum() / denom))
        active &= np.not_equal(arr, k)
    return hazards


def survival_from_delays(delays, max_steps):
    arr = np.asarray(delays, dtype=np.int64)
    out = np.zeros(max_steps, dtype=np.float64)
    for t in range(1, max_steps + 1):
        out[t - 1] = np.mean(arr > t)
    return out


def tv_distance(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(0.5 * np.abs(p - q).sum())


def mean_abs_diff_ignore_none(a, b):
    vals = []
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        vals.append(abs(float(x) - float(y)))
    return float(np.mean(vals)) if vals else 0.0


def load_model(spec, device):
    ckpt = torch.load(spec["ckpt"], map_location="cpu", weights_only=False)
    if spec["type"] == "vae_step":
        model = HandActionVAE(**_eval_old.infer_model_args(ckpt["model"])).to(device)
    elif spec["type"] == "chunk_vae":
        model = HandActionChunkVAE(**_eval_chunk.infer_model_args(ckpt["model"])).to(device)
    elif spec["type"] == "flow":
        model = HandActionChunkFlow(**_eval_flow.infer_model_args(ckpt)).to(device)
    elif spec["type"] == "state_space":
        model = HandActionStateSpaceChunkPrior(**_eval_ss.infer_model_args(ckpt)).to(device)
    else:
        raise ValueError(f"Unsupported type: {spec['type']}")
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def rollout_future_batch(spec, model, seed_window, num_rollouts, future_steps):
    device = next(model.parameters()).device
    seed_tensor = torch.from_numpy(seed_window).float().to(device)
    total_steps = model.window_size + future_steps
    actions = torch.zeros(num_rollouts, total_steps, model.action_dim, device=device)
    actions[:, : model.window_size, :] = seed_tensor.unsqueeze(0)

    if spec["type"] == "vae_step":
        t = model.window_size
        while t < total_steps:
            window = actions[:, t - model.window_size : t, :]
            pred = model.predict(window, deterministic=False)
            actions[:, t, :] = pred
            t += 1
        return actions[:, model.window_size :, :].cpu().numpy()

    if spec["type"] == "chunk_vae":
        t = model.window_size
        while t < total_steps:
            window = actions[:, t - model.window_size : t, :]
            pred = model.predict(window, deterministic=False)
            take = min(model.future_horizon, total_steps - t)
            actions[:, t : t + take, :] = pred[:, :take]
            t += take
        return actions[:, model.window_size :, :].cpu().numpy()

    if spec["type"] == "flow":
        t = model.window_size
        while t < total_steps:
            window = actions[:, t - model.window_size : t, :]
            pred = model.predict(
                window,
                num_integration_steps=spec.get("integration_steps", 24),
                return_latent=False,
            )
            take = min(model.future_horizon, total_steps - t)
            actions[:, t : t + take, :] = pred[:, :take]
            t += take
        return actions[:, model.window_size :, :].cpu().numpy()

    hidden = model.encode_history(seed_tensor.unsqueeze(0).expand(num_rollouts, -1, -1))
    current_state = actions[:, model.window_size - 1]
    prev_z = torch.zeros(num_rollouts, model.latent_dim, device=device)
    t = model.window_size
    while t < total_steps:
        pred, z, hidden, current_state = model.sample_step(
            hidden,
            current_state,
            prev_z,
            deterministic=False,
        )
        take = min(model.future_horizon, total_steps - t)
        actions[:, t : t + take, :] = pred[:, :take]
        prev_z = z
        t += take
    return actions[:, model.window_size :, :].cpu().numpy()


def collect_open_like_contexts(trajectories, window_size, open_eps, ref, horizon):
    rows = []
    open_anchor = ref["open_mean"][1:]
    for row in trajectories:
        actions = row["actions"]
        progress = progress_score(actions, ref)
        total = actions.shape[0]
        for t in range(total):
            hist = build_history_window(actions, t, window_size)
            if np.max(np.abs(hist[:, 1:] - open_anchor)) >= open_eps:
                continue
            gt_future = build_future(actions, t, horizon)
            rows.append(
                {
                    "traj_id": row["traj_id"],
                    "t": t,
                    "seed_window": hist,
                    "gt_future": gt_future,
                }
            )
    return rows


def compute_delay_distribution_from_contexts(contexts, ref, rise_threshold, horizon):
    delays = []
    for ctx in contexts:
        future_scores = progress_score(ctx["gt_future"], ref)
        delays.append(first_crossing(future_scores, rise_threshold))
    pmf = pmf_from_delays(delays, horizon)
    hazards = hazard_from_delays(delays, horizon)
    return {
        "num_contexts": int(len(delays)),
        "delays": delays,
        "close_prob": float(np.mean(np.asarray(delays) <= horizon)) if delays else 0.0,
        "delay_pmf": pmf.tolist(),
        "hazard": hazards,
    }


def compute_reset_reference(test_rows, ref, rise_threshold, max_steps, closed_tail, reopen_high, reopen_low):
    delays = []
    reopen_flags = []
    tail_progress = []
    tail_states = []
    for row in test_rows:
        actions = row["actions"]
        total = actions.shape[0]
        padded = np.concatenate(
            [actions, np.repeat(actions[-1:], max(0, max_steps - total), axis=0)],
            axis=0,
        )[:max_steps]
        scores = progress_score(padded, ref)
        delay = first_crossing(scores, rise_threshold)
        delays.append(delay)

        hi_idx = np.where(scores >= reopen_high)[0]
        if len(hi_idx) > 0:
            after = scores[hi_idx[0] + 1 :]
            reopen_flags.append(bool(np.any(after <= reopen_low)))
            tail_len = min(closed_tail, padded.shape[0])
            tail_progress.append(float(scores[-tail_len:].mean()))
            tail_states.append(padded[-tail_len:].mean(axis=0))
        else:
            reopen_flags.append(False)

    pmf = pmf_from_delays(delays, max_steps)
    hazards = hazard_from_delays(delays, max_steps)
    survival = survival_from_delays(delays, max_steps)
    tail_state_mean = np.stack(tail_states, axis=0).mean(axis=0) if tail_states else np.zeros(6, dtype=np.float32)
    return {
        "num_traj": int(len(test_rows)),
        "close_prob": float(np.mean(np.asarray(delays) <= max_steps)) if delays else 0.0,
        "delay_pmf": pmf.tolist(),
        "hazard": hazards,
        "survival": survival.tolist(),
        "reopen_rate": float(np.mean(reopen_flags)) if reopen_flags else 0.0,
        "closed_tail_progress_mean": float(np.mean(tail_progress)) if tail_progress else 0.0,
        "closed_tail_state_mean": tail_state_mean.tolist(),
    }


def evaluate_model_conditioned(spec, model, contexts, ref, rise_threshold, horizon, num_rollouts):
    all_delays = []
    for ctx in contexts:
        pred = rollout_future_batch(spec, model, ctx["seed_window"], num_rollouts, horizon)
        scores = progress_score(pred, ref)
        for sample_scores in scores:
            all_delays.append(first_crossing(sample_scores, rise_threshold))

    pmf = pmf_from_delays(all_delays, horizon)
    hazards = hazard_from_delays(all_delays, horizon)
    return {
        "num_samples": int(len(all_delays)),
        "close_prob": float(np.mean(np.asarray(all_delays) <= horizon)) if all_delays else 0.0,
        "delay_pmf": pmf.tolist(),
        "hazard": hazards,
    }


def evaluate_model_reset(spec, model, ref, rise_threshold, reopen_high, reopen_low, max_steps, num_rollouts, closed_tail):
    seed_state = ref["open_mean"].astype(np.float32)
    seed_window = np.repeat(seed_state[None, :], model.window_size, axis=0)
    pred = rollout_future_batch(spec, model, seed_window, num_rollouts, max_steps - model.window_size)
    prefix = np.repeat(seed_state[None, None, :], num_rollouts, axis=0)
    prefix = np.repeat(prefix, model.window_size, axis=1)
    full = np.concatenate([prefix, pred], axis=1)[:, :max_steps, :]

    scores = progress_score(full, ref)
    delays = []
    reopen_flags = []
    tail_progress = []
    tail_states = []
    for run, score in zip(full, scores):
        delay = first_crossing(score, rise_threshold)
        delays.append(delay)
        hi_idx = np.where(score >= reopen_high)[0]
        if len(hi_idx) > 0:
            after = score[hi_idx[0] + 1 :]
            reopen_flags.append(bool(np.any(after <= reopen_low)))
            tail_len = min(closed_tail, run.shape[0])
            tail_progress.append(float(score[-tail_len:].mean()))
            tail_states.append(run[-tail_len:].mean(axis=0))
        else:
            reopen_flags.append(False)

    pmf = pmf_from_delays(delays, max_steps)
    hazards = hazard_from_delays(delays, max_steps)
    survival = survival_from_delays(delays, max_steps)
    tail_state_mean = np.stack(tail_states, axis=0).mean(axis=0) if tail_states else np.zeros(6, dtype=np.float32)
    return {
        "num_rollouts": int(num_rollouts),
        "close_prob": float(np.mean(np.asarray(delays) <= max_steps)) if delays else 0.0,
        "delay_pmf": pmf.tolist(),
        "hazard": hazards,
        "survival": survival.tolist(),
        "reopen_rate": float(np.mean(reopen_flags)) if reopen_flags else 0.0,
        "closed_tail_progress_mean": float(np.mean(tail_progress)) if tail_progress else 0.0,
        "closed_tail_state_mean": tail_state_mean.tolist(),
    }


def summarize_against_reference(conditioned, reset, ref_conditioned, ref_reset):
    model_tail = np.asarray(reset["closed_tail_state_mean"], dtype=np.float64)
    ref_tail = np.asarray(ref_reset["closed_tail_state_mean"], dtype=np.float64)
    return {
        "open_like_close_prob": float(conditioned["close_prob"]),
        "open_like_close_prob_abs_err": abs(float(conditioned["close_prob"]) - float(ref_conditioned["close_prob"])),
        "open_like_delay_tv": tv_distance(conditioned["delay_pmf"], ref_conditioned["delay_pmf"]),
        "open_like_hazard_l1": mean_abs_diff_ignore_none(conditioned["hazard"], ref_conditioned["hazard"]),
        "reset_close_prob": float(reset["close_prob"]),
        "reset_close_prob_abs_err": abs(float(reset["close_prob"]) - float(ref_reset["close_prob"])),
        "reset_survival_l1": float(np.mean(np.abs(np.asarray(reset["survival"]) - np.asarray(ref_reset["survival"])))),
        "reset_reopen_rate": float(reset["reopen_rate"]),
        "closed_tail_progress_mean": float(reset["closed_tail_progress_mean"]),
        "closed_tail_progress_bias": float(reset["closed_tail_progress_mean"] - ref_reset["closed_tail_progress_mean"]),
        "closed_tail_joint_mae": float(np.mean(np.abs(model_tail - ref_tail))),
    }


def plot_summary(payload, out_path, lang):
    configure_plot_style(lang)
    gt_cond = payload["references"]["open_like_test"]
    gt_reset = payload["references"]["reset_test"]
    models = payload["models"]

    colors = plt.cm.tab10(np.linspace(0, 1, max(3, len(models))))
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    xs_cond = np.arange(1, payload["config"]["condition_horizon"] + 2)
    gt_pmf = np.asarray(gt_cond["delay_pmf"])
    axes[0, 0].bar(xs_cond, gt_pmf, width=0.65, color="#d0d7de", label=text(lang, "GT", "GT"))
    for color, model in zip(colors, models):
        axes[0, 0].plot(xs_cond, model["conditioned"]["delay_pmf"], marker="o", linewidth=2, label=model["label"], color=color)
    axes[0, 0].set_title(text(lang, "开放态条件下：首次离开 open 区的延迟分布", "Open-like conditioned first-leave delay PMF"))
    axes[0, 0].set_xlabel(text(lang, "延迟步数（最后一个柱子=H步内未触发）", "Delay step (last bin = no trigger within horizon)"))
    axes[0, 0].set_ylabel(text(lang, "概率", "Probability"))

    xs_h = np.arange(1, payload["config"]["condition_horizon"] + 1)
    axes[0, 1].plot(xs_h, gt_cond["hazard"], marker="o", linewidth=3, label=text(lang, "GT", "GT"), color="#111827")
    for color, model in zip(colors, models):
        hz = np.asarray([v if v is not None else np.nan for v in model["conditioned"]["hazard"]], dtype=np.float64)
        axes[0, 1].plot(xs_h, hz, marker="o", linewidth=2, label=model["label"], color=color)
    axes[0, 1].set_title(text(lang, "开放态条件下：每一步触发合手的离散 hazard", "Open-like conditioned discrete hazard"))
    axes[0, 1].set_xlabel(text(lang, "未来第 k 步", "Future step k"))
    axes[0, 1].set_ylabel(text(lang, "触发概率", "Trigger probability"))

    xs_surv = np.arange(1, payload["config"]["max_steps"] + 1)
    axes[1, 0].plot(xs_surv, gt_reset["survival"], linewidth=3, label=text(lang, "GT", "GT"), color="#111827")
    for color, model in zip(colors, models):
        axes[1, 0].plot(xs_surv, model["reset"]["survival"], linewidth=2, label=model["label"], color=color)
    axes[1, 0].set_title(text(lang, "从 reset 开始：仍保持 open 的生存曲线", "Reset survival curve (still open)"))
    axes[1, 0].set_xlabel(text(lang, "时间步", "Step"))
    axes[1, 0].set_ylabel(text(lang, "仍未离开 open 的概率", "Probability still in open basin"))

    labels = [m["label"] for m in models]
    x = np.arange(len(labels))
    reopen = [m["metrics"]["reset_reopen_rate"] for m in models]
    bias = [m["metrics"]["closed_tail_progress_bias"] for m in models]
    width = 0.38
    axes[1, 1].bar(x - width / 2, reopen, width=width, color="#e76f51", label=text(lang, "reopen 违例率", "reopen rate"))
    axes[1, 1].bar(x + width / 2, bias, width=width, color="#2a9d8f", label=text(lang, "闭合尾段均值偏差", "closed-tail bias"))
    axes[1, 1].axhline(0.0, color="#555555", linewidth=1.2)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1, 1].set_title(text(lang, "长滚动约束：reopen 与闭合幅度校准", "Long-rollout constraints: reopen and closure calibration"))
    axes[1, 1].set_ylabel(text(lang, "数值", "Value"))

    for ax in axes.flat:
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=9)
    axes[0, 1].legend(fontsize=9)
    axes[1, 0].legend(fontsize=9)
    axes[1, 1].legend(fontsize=9)

    fig.suptitle(text(lang, "手部状态先验：分布型评估总览", "Hand prior: distributional evaluation summary"), fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    test_dir = Path(args.test_dir)
    train_dir = test_dir.parent / "train"

    model_specs = load_model_specs(args.models_json)
    test_rows = load_trajectories(test_dir)
    ref = load_progress_reference(train_dir, closed_tail=args.closed_tail, move_threshold=args.closed_move_threshold)

    open_context_cache = {}
    open_ref_cache = {}
    first_window_size = None
    for spec in model_specs:
        model = load_model(spec, device)
        win = int(model.window_size)
        if first_window_size is None:
            first_window_size = win
        if win not in open_context_cache:
            contexts = collect_open_like_contexts(
                test_rows,
                window_size=win,
                open_eps=args.open_eps,
                ref=ref,
                horizon=args.condition_horizon,
            )
            open_context_cache[win] = contexts
            open_ref_cache[win] = compute_delay_distribution_from_contexts(
                contexts,
                ref=ref,
                rise_threshold=args.rise_threshold,
                horizon=args.condition_horizon,
            )

    reset_ref = compute_reset_reference(
        test_rows,
        ref=ref,
        rise_threshold=args.rise_threshold,
        max_steps=args.max_steps,
        closed_tail=args.closed_tail,
        reopen_high=args.reopen_high,
        reopen_low=args.reopen_low,
    )

    payload = {
        "config": {
            "test_dir": str(test_dir),
            "train_dir": str(train_dir),
            "condition_horizon": int(args.condition_horizon),
            "condition_rollouts": int(args.condition_rollouts),
            "reset_rollouts": int(args.reset_rollouts),
            "max_steps": int(args.max_steps),
            "open_eps": float(args.open_eps),
            "rise_threshold": float(args.rise_threshold),
            "reopen_high": float(args.reopen_high),
            "reopen_low": float(args.reopen_low),
            "closed_tail": int(args.closed_tail),
            "seed": int(args.seed),
        },
        "references": {
            "progress_reference": {
                "open_mean": ref["open_mean"].tolist(),
                "closed_mean": ref["closed_mean"].tolist(),
            },
            "open_like_test": open_ref_cache[first_window_size],
            "reset_test": reset_ref,
        },
        "models": [],
    }

    for spec in model_specs:
        print(f"[distributional-eval] {spec['key']}")
        model = load_model(spec, device)
        contexts = open_context_cache[int(model.window_size)]
        ref_conditioned = open_ref_cache[int(model.window_size)]
        conditioned = evaluate_model_conditioned(
            spec,
            model,
            contexts=contexts,
            ref=ref,
            rise_threshold=args.rise_threshold,
            horizon=args.condition_horizon,
            num_rollouts=args.condition_rollouts,
        )
        reset = evaluate_model_reset(
            spec,
            model,
            ref=ref,
            rise_threshold=args.rise_threshold,
            reopen_high=args.reopen_high,
            reopen_low=args.reopen_low,
            max_steps=args.max_steps,
            num_rollouts=args.reset_rollouts,
            closed_tail=args.closed_tail,
        )
        metrics = summarize_against_reference(conditioned, reset, ref_conditioned, reset_ref)
        payload["models"].append(
            {
                "key": spec["key"],
                "label": spec["label"],
                "type": spec["type"],
                "family": spec.get("family", "unknown"),
                "ckpt": spec["ckpt"],
                "conditioned": conditioned,
                "reset": reset,
                "metrics": metrics,
            }
        )

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if args.output_plot:
        plot_summary(payload, args.output_plot, args.lang)

    print(json.dumps({"output_json": args.output_json, "output_plot": args.output_plot}, ensure_ascii=False))


if __name__ == "__main__":
    main()
