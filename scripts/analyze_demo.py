"""
Diagnostic metrics for the delta-action Hand Action VAE sweep.

Computes per-checkpoint:
  0. PCA explained variance on flattened (8, 6) windows  — linear upper bound
  A. Latent linear-decodability R²                       — VAE encoder vs PCA
  B. Active latent dim count (per-dim KL > 0.01)         — utilization
  C. GT comparison per-step MSE on test trajectories     — AR fidelity
  D. Decoder reachability (optimal-z MSE)                — decoder ceiling
"""

import importlib.util
from pathlib import Path

import numpy as np
import torch

# ───────────────────────────── Config ─────────────────────────────

DATA_ROOT = Path("/home/yxy/VQ-VLA/data/delta_action_20260327_11_10_43")
WINDOW_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT = Path("outputs/delta")

CHECKPOINTS = {
    # ── Sweep 1: Architecture ──
    "A  h256 d1 mlp (baseline)":      OUT / "arch_sweep/A_h256_d1_mlp/checkpoint.pth",
    "B  h512 d1 mlp":                 OUT / "arch_sweep/B_h512_d1_mlp/checkpoint.pth",
    "C  h256 d2 mlp":                 OUT / "arch_sweep/C_h256_d2_mlp/checkpoint.pth",
    "D  h512 d2 mlp":                 OUT / "arch_sweep/D_h512_d2_mlp/checkpoint.pth",
    "E  h256 d1 cnn":                 OUT / "arch_sweep/E_h256_d1_cnn/checkpoint.pth",
    # ── Sweep 2: Loss ──
    "F  recon_aux=0.1":               OUT / "loss_sweep/F_recon_only/checkpoint.pth",
    "G  free_bits=0.5":               OUT / "loss_sweep/G_freebits_only/checkpoint.pth",
    "H  both":                        OUT / "loss_sweep/H_both/checkpoint.pth",
    # ── Sweep 3: Recon weight ──
    "W005  recon_aux=0.05":           OUT / "recon_sweep/W005/checkpoint.pth",
    "W030  recon_aux=0.30":           OUT / "recon_sweep/W030/checkpoint.pth",
    "W050  recon_aux=0.50":           OUT / "recon_sweep/W050/checkpoint.pth",
    "W100  recon_aux=1.00":           OUT / "recon_sweep/W100/checkpoint.pth",
}

# ───────────────────────────── Loading ─────────────────────────────

def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_vae_root = Path(__file__).parent.parent / "vae"
HandActionVAE = _load_module(_vae_root / "model/hand_vae.py", "hand_vae").HandActionVAE
infer_model_args = _load_module(_vae_root / "scripts/eval.py", "eval_mod").infer_model_args


def load_trajectories(split_dir: Path):
    """Return list of (T, 6) numpy arrays — one per trajectory."""
    files = sorted(split_dir.glob("trajectory_*_demo_expert.pt"))
    trajs = []
    for f in files:
        data = torch.load(f, map_location="cpu", weights_only=False)
        trajs.append(data["actions"][:, 0, 6:12].float().numpy())
    return trajs


def build_windows(trajs, window_size):
    """Build flat / sequence windows + their corresponding next-frame targets."""
    flat_windows, seq_windows, targets = [], [], []
    for actions in trajs:
        T = actions.shape[0]
        for t in range(T):
            start = t - window_size + 1
            if start < 0:
                # Zero-pad for delta data (auto-detect already done in dataset;
                # here we hard-code zero pad since this script targets delta data)
                pad = np.zeros((-start, actions.shape[1]))
                window = np.concatenate([pad, actions[0:t + 1]], axis=0)
            else:
                window = actions[start:t + 1]
            seq_windows.append(window)
            flat_windows.append(window.flatten())
            targets.append(actions[t + 1] if t + 1 < T else actions[t])
    return np.stack(flat_windows), np.stack(seq_windows), np.stack(targets)


def load_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    kwargs = infer_model_args(ckpt["model"])
    model = HandActionVAE(**kwargs)
    model.load_state_dict(ckpt["model"])
    model.eval().to(DEVICE)
    return model, kwargs


# ───────────────────────────── Metrics ─────────────────────────────

def pca_explained(X: np.ndarray):
    """Per-component explained variance ratio (descending)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / (Xc.shape[0] - 1)
    return var / var.sum()


@torch.no_grad()
def encode_all(model, seq_windows: np.ndarray, batch_size=512):
    """Encode all windows. Returns (mu, log_var) as numpy."""
    mus, lvs = [], []
    for i in range(0, len(seq_windows), batch_size):
        batch = torch.from_numpy(seq_windows[i:i + batch_size]).float().to(DEVICE)
        mu, lv = model.encode(batch)
        mus.append(mu.cpu().numpy())
        lvs.append(lv.cpu().numpy())
    return np.concatenate(mus), np.concatenate(lvs)


def linear_decodability_r2(mu: np.ndarray, X: np.ndarray):
    """R² of best linear regression mu (N, k) → X (N, D)."""
    mu_c = mu - mu.mean(axis=0, keepdims=True)
    X_c = X - X.mean(axis=0, keepdims=True)
    W, _, _, _ = np.linalg.lstsq(mu_c, X_c, rcond=None)
    X_pred = mu_c @ W
    ss_res = np.sum((X_c - X_pred) ** 2)
    ss_tot = np.sum(X_c ** 2)
    return 1 - ss_res / ss_tot


def active_dims(mu: np.ndarray, log_var: np.ndarray, threshold=0.01):
    kl_per_dim = -0.5 * (1 + log_var - mu ** 2 - np.exp(log_var))
    kl_per_dim = kl_per_dim.mean(axis=0)
    return kl_per_dim, int((kl_per_dim > threshold).sum())


@torch.no_grad()
def actual_val_recon_per_target(model, seq_windows, targets, batch_size=512):
    out = []
    for i in range(0, len(seq_windows), batch_size):
        x = torch.from_numpy(seq_windows[i:i + batch_size]).float().to(DEVICE)
        t = torch.from_numpy(targets[i:i + batch_size]).float().to(DEVICE)
        mu, _ = model.encode(x)
        pred = model.decode(mu)
        out.append(((pred - t) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(out)


def decoder_reachability(model, target_actions, n_iter=600, lr=0.05,
                         batch_size=512, n_restarts=8):
    saved = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        targets_t = torch.from_numpy(target_actions).float().to(DEVICE)
        N = targets_t.shape[0]
        latent_dim = model.latent_dim
        all_min_mse = []
        for i in range(0, N, batch_size):
            batch = targets_t[i:i + batch_size]
            best_mse = None
            for restart in range(n_restarts):
                spread = [0.1, 0.5, 1.0, 1.5, 2.0, 0.5, 1.0, 0.3][restart % 8]
                z = (torch.randn(batch.shape[0], latent_dim, device=DEVICE) * spread).requires_grad_(True)
                opt = torch.optim.Adam([z], lr=lr)
                for _ in range(n_iter):
                    opt.zero_grad()
                    pred = model.decode(z)
                    loss = ((pred - batch) ** 2).sum()
                    loss.backward()
                    opt.step()
                with torch.no_grad():
                    pred_final = model.decode(z)
                    per_sample_mse = ((pred_final - batch) ** 2).mean(dim=1)
                best_mse = per_sample_mse if best_mse is None else torch.minimum(best_mse, per_sample_mse)
            all_min_mse.append(best_mse.cpu().numpy())
        return np.concatenate(all_min_mse)
    finally:
        for p, r in zip(model.parameters(), saved):
            p.requires_grad_(r)


@torch.no_grad()
def gt_per_step_mse(model, trajs, window_size, n_samples=20):
    """Batched AR rollout: per-step MSE vs GT."""
    all_mses = []
    for actions in trajs:
        T = actions.shape[0]
        if T <= window_size:
            continue
        gt = torch.from_numpy(actions).float().to(DEVICE)
        seed = gt[:window_size].unsqueeze(0).expand(n_samples, -1, -1).clone()
        preds = [seed[:, i, :] for i in range(window_size)]
        for _ in range(window_size, T):
            window = torch.stack(preds[-window_size:], dim=1)
            next_pred = model.predict(window, deterministic=False)
            preds.append(next_pred)
        traj_pred = torch.stack(preds, dim=1)
        gt_b = gt.unsqueeze(0).expand(n_samples, -1, -1)
        mse = ((traj_pred - gt_b) ** 2).mean(dim=2).mean(dim=0)
        all_mses.append(mse.cpu().numpy())
    max_len = max(len(m) for m in all_mses)
    padded = np.full((len(all_mses), max_len), np.nan)
    for i, m in enumerate(all_mses):
        padded[i, :len(m)] = m
    return np.nanmean(padded, axis=0)


# ───────────────────────────── Main ─────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    train_dir = DATA_ROOT / "train"
    test_dir = DATA_ROOT / "test"

    print("Loading delta-action trajectories...")
    train_trajs = load_trajectories(train_dir)
    test_trajs = load_trajectories(test_dir)
    train_flat, train_seq, train_targets = build_windows(train_trajs, WINDOW_SIZE)
    test_flat, test_seq, test_targets = build_windows(test_trajs, WINDOW_SIZE)
    print(f"  train: {len(train_trajs)} trajs -> {train_flat.shape[0]} windows")
    print(f"  test:  {len(test_trajs)} trajs -> {test_flat.shape[0]} windows")

    # PCA bounds
    print("\n" + "=" * 70)
    print("[0] PCA on flattened (8, 6) windows -- linear upper bound")
    print("=" * 70)
    ratios = pca_explained(train_flat)
    cum = np.cumsum(ratios) * 100
    for k in [1, 2, 3, 4, 8, 16, 32, 48]:
        if k <= len(ratios):
            print(f"  top-{k:2d}: {cum[k-1]:6.3f}%")

    # Per-model metrics
    summary = []
    for label, ckpt_path in CHECKPOINTS.items():
        print("\n" + "=" * 70)
        print(f"Model: {label}")
        print(f"  ckpt: {ckpt_path}")
        print("=" * 70)
        if not ckpt_path.exists():
            print("  [skip] checkpoint not found")
            continue

        model, kwargs = load_model(ckpt_path)
        k = kwargs["latent_dim"]
        print(f"  hidden={kwargs['hidden_dim']}, latent={k}, encoder={kwargs['encoder_type']}")

        mu_tr, lv_tr = encode_all(model, train_seq)
        print(f"  encoded {mu_tr.shape[0]} train windows -> mu {mu_tr.shape}")

        # [A] Linear decodability vs PCA bound
        r2_tr = linear_decodability_r2(mu_tr, train_flat) * 100
        pca_bound = cum[k - 1]
        print(f"\n  [A] Linear decodability R² vs PCA top-{k} bound = {pca_bound:.2f}%")
        print(f"      VAE encoder R² (train): {r2_tr:6.3f}%   gap: {pca_bound - r2_tr:+.2f} pts")

        # [B] Active dims
        kl_per_dim, n_active = active_dims(mu_tr, lv_tr, threshold=0.01)
        print(f"\n  [B] Active latent dims (KL > 0.01):")
        print(f"      Active: {n_active} / {k}")
        if k <= 8:
            print(f"      Per-dim KL: {np.round(kl_per_dim, 4).tolist()}")

        # [C] GT per-step MSE
        per_step = gt_per_step_mse(model, test_trajs, WINDOW_SIZE, n_samples=20)
        avg_mse = float(np.nanmean(per_step))
        early = float(np.nanmean(per_step[WINDOW_SIZE:WINDOW_SIZE + 5]))
        mid = float(np.nanmean(per_step[WINDOW_SIZE + 5:WINDOW_SIZE + 15]))
        late = float(np.nanmean(per_step[WINDOW_SIZE + 15:]))
        print(f"\n  [C] GT rollout per-step MSE on {len(test_trajs)} test trajs (20 samples each):")
        print(f"      mean over horizon:  {avg_mse:.6f}")
        print(f"      early  (steps 8-12):  {early:.6f}")
        print(f"      mid    (steps 13-22): {mid:.6f}")
        print(f"      late   (steps 23+):   {late:.6f}")

        # [D] Decoder reachability
        actual_per_target = actual_val_recon_per_target(model, test_seq, test_targets)
        actual_mean = float(actual_per_target.mean())
        floor_per_target = decoder_reachability(model, test_targets, n_iter=600, lr=0.05, n_restarts=8)
        floor_mean = float(floor_per_target.mean())
        floor_median = float(np.median(floor_per_target))
        floor_p95 = float(np.quantile(floor_per_target, 0.95))
        pairwise_ratio = floor_per_target / np.maximum(actual_per_target, 1e-9)
        frac_dec_better = float((floor_per_target < actual_per_target * 0.5).mean())
        median_ratio = float(np.median(pairwise_ratio))

        print(f"\n  [D] Decoder reachability -- multi-restart optimal z*:")
        print(f"      Actual recon  (encoder mu): mean={actual_mean:.6f}")
        print(f"      Decoder floor (best z*):    mean={floor_mean:.6f}, median={floor_median:.6f}, p95={floor_p95:.6f}")
        print(f"      Per-target median ratio (floor/actual): {median_ratio:.3f}")
        print(f"      Fraction where decoder could do >=2x better: {frac_dec_better*100:.1f}%")

        if floor_median < actual_mean * 0.05:
            verdict = "DECODER fine -- ENCODER bottleneck"
        elif floor_median < actual_mean * 0.3:
            verdict = "DECODER mostly fine -- encoder is main bottleneck"
        elif floor_median > actual_mean * 0.7:
            verdict = "DECODER weak -- needs more capacity"
        else:
            verdict = "MIXED -- both encoder and decoder contribute"
        print(f"      -> Verdict: {verdict}")

        summary.append({
            "label": label, "k": k, "r2": r2_tr, "pca_bound": pca_bound,
            "n_active": n_active, "ar_mse": avg_mse,
            "actual_recon": actual_mean, "floor_median": floor_median,
            "frac_dec_better": frac_dec_better, "verdict": verdict,
        })

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY (delta-action sweep)")
    print("=" * 70)
    print(f"{'model':<32} {'k':>3} {'R2':>7} {'active':>8} {'AR MSE':>10} "
          f"{'enc recon':>10} {'dec floor med':>14}")
    for s in summary:
        print(f"{s['label']:<32} {s['k']:>3}  "
              f"{s['r2']:>5.2f}%  "
              f"{s['n_active']:>3}/{s['k']:<3}   "
              f"{s['ar_mse']:.6f} "
              f"{s['actual_recon']:.6f}    "
              f"{s['floor_median']:.6f}")
        print(f"{'':<32} -> {s['verdict']}")


if __name__ == "__main__":
    main()
