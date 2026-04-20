"""MoA Prediction: Transductive SVD + warm-init LogReg + vec_2layer ensemble
Building on chanbin-test-moa/af3682d5 (-0.01539)
Design:
  - TruncatedSVD n_iter=1 (fast: ~18s vs 138s for PCA default)
  - Stat features + combined SVD + cross-PCA
  - vec_logreg n_iter=60: fast convergence via warm bias init (logit of base rate)
  - Nystroem RBF kernel for non-linearity
  - vec_2layer: vectorized 2-layer MLP with warm bias, He init, Adam
    avoids sklearn MLPClassifier slowness on 99.7% sparse multi-label data
  - Per-target adaptive Bayesian calibration
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.kernel_approximation import Nystroem
import time
import warnings
warnings.filterwarnings('ignore')

t0 = time.time()

# === Load data ===
train_features = pd.read_csv("data/train_features.csv")
train_targets  = pd.read_csv("data/train_targets.csv")
test_features  = pd.read_csv("data/test_features.csv")
target_cols = [c for c in train_targets.columns if c != "sig_id"]
n_tar = len(target_cols)

ctrl_tr = (train_features["cp_type"] == "ctl_vehicle").values
ctrl_te = (test_features["cp_type"] == "ctl_vehicle").values
Xtr = train_features[~ctrl_tr].reset_index(drop=True)
ytr = train_targets[~ctrl_tr][target_cols].values.astype(np.float32)
Xte = test_features[~ctrl_te].reset_index(drop=True)

gene_cols = [c for c in train_features.columns if c.startswith("g-")]
cell_cols = [c for c in train_features.columns if c.startswith("c-")]
for df in [Xtr, Xte]:
    df["cp_t"] = df["cp_time"].map({24: 0.0, 48: 0.5, 72: 1.0})
    df["cp_d"] = df["cp_dose"].map({"D1": 0.0, "D2": 1.0})


def kurtosis(arr):
    m = arr.mean(axis=1, keepdims=True)
    v = ((arr - m) ** 2).mean(axis=1, keepdims=True).clip(1e-12)
    return (((arr - m) ** 4).mean(axis=1) / v.squeeze() ** 2) - 3


def stat_block(arr):
    p5, p25, p75, p95 = np.percentile(arr, [5, 25, 75, 95], axis=1)
    return np.column_stack([p5, p25, p75, p95, p75 - p25, kurtosis(arr)]).astype(np.float32)


def build_features(train_df, test_df):
    n_tr = len(train_df)
    tr_g = train_df[gene_cols].values.astype(np.float32)
    te_g = test_df[gene_cols].values.astype(np.float32)
    tr_c = train_df[cell_cols].values.astype(np.float32)
    te_c = test_df[cell_cols].values.astype(np.float32)
    all_g  = np.vstack([tr_g, te_g])
    all_c  = np.vstack([tr_c, te_c])
    all_gc = np.hstack([all_g, all_c])
    ag  = StandardScaler().fit_transform(all_g)
    ac  = StandardScaler().fit_transform(all_c)
    agc = StandardScaler().fit_transform(all_gc)
    svg  = TruncatedSVD(100, n_iter=1, random_state=0).fit_transform(ag)
    svc  = TruncatedSVD(50, n_iter=1, random_state=0).fit_transform(ac)
    svgc = TruncatedSVD(40, n_iter=1, random_state=0).fit_transform(agc)
    # Cross-PCA: SVD on gene×cell outer products captures interaction manifold
    n_all = len(all_g)
    cross_flat = (svg[:, :25, None] * svc[:, None, :20]).reshape(n_all, -1)
    cross_pca = TruncatedSVD(25, n_iter=1, random_state=0).fit_transform(cross_flat)
    cp_t = np.concatenate([train_df["cp_t"].values, test_df["cp_t"].values])
    cp_d = np.concatenate([train_df["cp_d"].values, test_df["cp_d"].values])
    X = np.hstack([
        svg, svc, svgc,
        svg[:, :25] * cp_t[:, None], svg[:, :25] * cp_d[:, None],
        svc[:, :12] * cp_t[:, None], svc[:, :12] * cp_d[:, None],
        svg[:, :10] * svc[:, :10],
        cross_pca,
        stat_block(all_g), stat_block(all_c),
        np.column_stack([cp_t, cp_d, cp_t * cp_d, cp_t ** 2]),
    ]).astype(np.float32)
    return X[:n_tr], X[n_tr:]


print("Building features...")
X_tr, X_te = build_features(Xtr, Xte)
print(f"  train={X_tr.shape}  test={X_te.shape}  t={time.time()-t0:.1f}s")

sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr).astype(np.float32)
X_te_s = sc.transform(X_te).astype(np.float32)

base_rates = ytr.mean(axis=0)
has_pos    = (ytr.sum(axis=0) > 0)
valid_idx  = np.where(has_pos)[0]
n_valid    = int(has_pos.sum())
ytr_v      = ytr[:, has_pos]
print(f"Trainable targets: {n_valid}/{n_tar}")


def full_pred(p_valid):
    out = np.full((len(p_valid), n_tar), 0.001, dtype=np.float32)
    out[:, valid_idx] = p_valid
    return out


def vec_logreg(X, y, C=0.1, n_iter=60, lr=0.01):
    """Vectorized multi-output LogReg + Adam + warm bias init."""
    n, p = X.shape; m = y.shape[1]
    Xb = np.hstack([X, np.ones((n, 1), dtype=np.float32)])
    base = y.mean(0).clip(1e-4, 1 - 1e-4).astype(np.float32)
    W = np.zeros((p + 1, m), dtype=np.float32)
    W[-1] = np.log(base / (1.0 - base))
    XtY = (Xb.T @ y).astype(np.float32)
    reg = np.float32(1.0 / (C * n))
    b1, b2, eps = np.float32(0.9), np.float32(0.999), np.float32(1e-8)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    for it in range(1, n_iter + 1):
        pred = 1.0 / (1.0 + np.exp(-np.clip(Xb @ W, -50, 50)))
        g = (Xb.T @ pred - XtY) / n
        g[:-1] += reg * W[:-1]
        mW = b1 * mW + (1 - b1) * g
        vW = b2 * vW + (1 - b2) * (g * g)
        W -= lr * (mW / (1 - b1 ** it)) / (np.sqrt(vW / (1 - b2 ** it)) + eps)
    def predict(Xnew):
        Xb2 = np.hstack([Xnew, np.ones((len(Xnew), 1), dtype=np.float32)])
        return 1.0 / (1.0 + np.exp(-np.clip(Xb2 @ W, -50, 50)))
    return predict


def vec_2layer(X, y, C=0.1, hidden=64, n_iter=80, lr=0.005, seed=0):
    """Vectorized 2-layer MLP: X → hidden → ReLU → 205 outputs.
    All targets trained together. Warm bias = logit(base_rate) for fast
    convergence on 99.7%-sparse targets without sklearn overhead.
    """
    n, p = X.shape; m = y.shape[1]
    rng = np.random.default_rng(seed)
    W1 = (rng.standard_normal((p, hidden)) * np.sqrt(2.0 / p)).astype(np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    W2 = (rng.standard_normal((hidden, m)) * np.sqrt(2.0 / hidden)).astype(np.float32)
    base = y.mean(0).clip(1e-4, 1 - 1e-4).astype(np.float32)
    b2 = np.log(base / (1.0 - base))
    reg = np.float32(1.0 / (C * n))
    b1a, b2a, eps = np.float32(0.9), np.float32(0.999), np.float32(1e-8)
    mW1, vW1 = np.zeros_like(W1), np.zeros_like(W1)
    mb1, vb1 = np.zeros_like(b1), np.zeros_like(b1)
    mW2, vW2 = np.zeros_like(W2), np.zeros_like(W2)
    mb2, vb2 = np.zeros_like(b2), np.zeros_like(b2)
    for it in range(1, n_iter + 1):
        bc, bv = 1.0 - b1a ** it, 1.0 - b2a ** it
        h    = np.maximum(0.0, X @ W1 + b1)
        pred = 1.0 / (1.0 + np.exp(-np.clip(h @ W2 + b2, -50, 50)))
        err  = (pred - y) / n
        gW2 = h.T @ err;   gW2 += reg * W2
        gb2 = err.sum(0)
        dh  = (err @ W2.T) * (h > 0)
        gW1 = X.T @ dh;    gW1 += reg * W1
        gb1 = dh.sum(0)
        for param, g, mP, vP in [(W2,gW2,mW2,vW2),(b2,gb2,mb2,vb2),
                                   (W1,gW1,mW1,vW1),(b1,gb1,mb1,vb1)]:
            mP[:] = b1a * mP + (1 - b1a) * g
            vP[:] = b2a * vP + (1 - b2a) * (g * g)
            param -= lr * (mP / bc) / (np.sqrt(vP / bv) + eps)
    def predict(Xnew):
        h = np.maximum(0.0, Xnew @ W1 + b1)
        return 1.0 / (1.0 + np.exp(-np.clip(h @ W2 + b2, -50, 50)))
    return predict


from scipy.optimize import minimize as scipy_minimize

all_preds   = []   # test predictions from each model
all_val     = []   # val predictions (for OOF weight optimization)
model_cfgs  = []   # config dicts for two-pass retraining on full data

# OOF split: 5% of training data for blend weight optimization
# (small holdout = more training data; weight estimation still stable
#  because optimizer works over 205 × n_val scalar pairs per model)
VAL_FRAC = 0.05
np.random.seed(0)
_perm      = np.random.permutation(len(X_tr_s))
_n_val     = int(len(X_tr_s) * VAL_FRAC)
_val_idx   = _perm[:_n_val]
_tr_idx    = _perm[_n_val:]
X_tr_oof   = X_tr_s[_tr_idx];    X_val_oof  = X_tr_s[_val_idx]
y_tr_oof   = ytr_v[_tr_idx];     y_val_oof  = ytr_v[_val_idx]
print(f"OOF split: {len(X_tr_oof)} train / {len(X_val_oof)} val")

# Budget: reserve 40s for OOF weight opt + ensemble/write
BUDGET = 260.0
tr = lambda: BUDGET - (time.time() - t0)

def _add(test_pv, val_pv, cfg=None):
    all_preds.append(full_pred(np.clip(test_pv, 0, 1).astype(np.float32)))
    all_val.append(full_pred(np.clip(val_pv,  0, 1).astype(np.float32)))
    model_cfgs.append(cfg)  # None = can't retrain (nystroem/modality)

# --- Ridge ---
print("Ridge...")
for alpha in [30.0, 100.0, 500.0]:
    t1 = time.time()
    r = Ridge(alpha=alpha, fit_intercept=True)
    r.fit(X_tr_oof, y_tr_oof)
    _add(r.predict(X_te_s), r.predict(X_val_oof), {'type': 'ridge', 'alpha': alpha})
    print(f"  alpha={alpha}: {time.time()-t1:.1f}s")

# --- Vectorized LogReg ensemble ---
print("LogReg ensemble...")
for C in [0.03, 0.07, 0.15, 0.30, 0.60, 1.0, 2.0]:
    if tr() < 8:
        break
    t1 = time.time()
    n_it = 100 if C >= 0.30 else 80  # weakly-regularized needs more iters
    fn = vec_logreg(X_tr_oof, y_tr_oof, C=C, n_iter=n_it, lr=0.008)
    _add(fn(X_te_s), fn(X_val_oof), {'type': 'logreg', 'C': C, 'n_iter': n_it, 'lr': 0.008})
    print(f"  C={C} n={n_it}: {time.time()-t1:.1f}s  total={time.time()-t0:.1f}s")

# --- Modality-specific LogReg (gene-only and cell-only SVD features) ---
# svg components = cols 0:80, svc components = cols 80:120
print("Modality LogReg...")
for X_mod, label in [(X_tr_oof[:, :80], "gene"), (X_tr_oof[:, 80:120], "cell")]:
    X_mod_val = X_val_oof[:, :80] if label == "gene" else X_val_oof[:, 80:120]
    X_mod_te  = X_te_s[:, :80] if label == "gene" else X_te_s[:, 80:120]
    for C in [0.1, 0.5]:
        if tr() < 15:
            print(f"  Skipping {label} C={C}, {tr():.0f}s remain")
            break
        t1 = time.time()
        fn = vec_logreg(X_mod, y_tr_oof, C=C, n_iter=60, lr=0.01)
        _add(fn(X_mod_te), fn(X_mod_val))
        print(f"  {label} C={C}: {time.time()-t1:.1f}s  total={time.time()-t0:.1f}s")

# --- Nystroem RBF kernel + LogReg ---
if tr() > 90:
    print("Nystroem + LogReg...")
    t1 = time.time()
    ny = Nystroem(kernel='rbf', n_components=150, random_state=0)
    ny.fit(X_tr_oof)
    sc_ny = StandardScaler()
    X_tr_ny  = sc_ny.fit_transform(ny.transform(X_tr_oof).astype(np.float32))
    X_val_ny = sc_ny.transform(ny.transform(X_val_oof).astype(np.float32))
    X_te_ny  = sc_ny.transform(ny.transform(X_te_s).astype(np.float32))
    print(f"  Nystroem transform: {time.time()-t1:.1f}s")
    for C in [0.1, 0.3, 0.7]:
        if tr() < 12:
            break
        t1 = time.time()
        fn = vec_logreg(X_tr_ny, y_tr_oof, C=C, n_iter=40, lr=0.01)
        _add(fn(X_te_ny), fn(X_val_ny))
        print(f"  Ny+LR C={C}: {time.time()-t1:.1f}s  total={time.time()-t0:.1f}s")
else:
    print(f"Skipping Nystroem, {tr():.0f}s remain")

# --- Modality-specific MLPs (gene-only / cell-only, fast) ---
# svg=cols 0:100, svc=cols 100:150; these are fast because input dim << 315
if tr() > 20:
    print("Modality MLPs...")
    for feat_slice, label in [
        (slice(0, 100),  "gene"),
        (slice(100, 150), "cell"),
    ]:
        for seed in range(3):
            if tr() < 10:
                break
            t1 = time.time()
            fn = vec_2layer(X_tr_oof[:, feat_slice], y_tr_oof, C=0.10, hidden=64,
                            n_iter=80, lr=0.005, seed=seed)
            _add(fn(X_te_s[:, feat_slice]), fn(X_val_oof[:, feat_slice]))
            print(f"  MLP-{label} s={seed}: {time.time()-t1:.1f}s  total={time.time()-t0:.1f}s")

# --- Vectorized 2-layer MLP ensemble ---
print("2-layer MLP ensemble...")
mlp2_configs = [
    # h=64 first: fast models run even on loaded systems; h=128 appended at end
    # so budget guard gracefully skips h=128 when system is slow
    (0,  64,  60, 0.005, 0.10),
    (1,  64,  60, 0.005, 0.10),
    (2,  64,  60, 0.005, 0.10),
    (3,  64,  60, 0.005, 0.10),
    (4,  64,  60, 0.005, 0.10),
    (5,  64,  60, 0.005, 0.10),
    (6,  64,  60, 0.005, 0.10),
    (7,  64,  60, 0.005, 0.10),
    (8,  64,  60, 0.005, 0.10),
    (9,  64,  60, 0.005, 0.10),
    (0,  64,  60, 0.005, 0.20),
    (1,  64,  60, 0.005, 0.20),
    (2,  64,  60, 0.005, 0.20),
    (3,  64,  60, 0.005, 0.20),
    (4,  64,  60, 0.005, 0.20),
    (5,  64,  60, 0.005, 0.20),
    # h=64 80-iter: better convergence, still fast
    (0,  64,  80, 0.003, 0.10),
    (1,  64,  80, 0.003, 0.10),
    (2,  64,  80, 0.003, 0.10),
    (3,  64,  80, 0.003, 0.10),
    # h=128: higher capacity, only runs when budget permits
    (0,  128, 50, 0.003, 0.10),
    (1,  128, 50, 0.003, 0.10),
    (2,  128, 50, 0.003, 0.10),
    (3,  128, 50, 0.003, 0.10),
    (4,  128, 50, 0.003, 0.10),
    (0,  128, 50, 0.003, 0.20),
    (1,  128, 50, 0.003, 0.20),
    (2,  128, 50, 0.003, 0.20),
    # h=256: highest capacity (chanbin's 'wider MLPs'); ~2-15s each depending
    # on system load; only runs when budget permits (>65s remaining)
    (0,  256, 35, 0.003, 0.10),
    (1,  256, 35, 0.003, 0.10),
    (2,  256, 35, 0.003, 0.10),
    (3,  256, 35, 0.003, 0.10),
    (4,  256, 35, 0.003, 0.10),
    (5,  256, 35, 0.003, 0.10),
    (6,  256, 35, 0.003, 0.10),
    (0,  256, 35, 0.003, 0.20),
    (1,  256, 35, 0.003, 0.20),
    (2,  256, 35, 0.003, 0.20),
    (3,  256, 35, 0.003, 0.20),
    # h=256 more iters: better convergence for the wider network
    (0,  256, 60, 0.002, 0.10),
    (1,  256, 60, 0.002, 0.10),
    (2,  256, 60, 0.002, 0.10),
    (0,  256, 60, 0.002, 0.20),
    (1,  256, 60, 0.002, 0.20),
    # h=256 strong regularization: helps with 99.7% sparse targets
    (0,  256, 50, 0.003, 0.05),
    (1,  256, 50, 0.003, 0.05),
    (2,  256, 50, 0.003, 0.05),
]
for seed, hidden, n_iter, lr, C in mlp2_configs:
    remaining = tr()
    if remaining < 25:
        print(f"  Stopping, {remaining:.0f}s remain")
        break
    if hidden >= 128 and remaining < 45:
        continue
    t1 = time.time()
    print(f"  2L h={hidden} s={seed} C={C}  t={time.time()-t0:.1f}s ...")
    fn = vec_2layer(X_tr_oof, y_tr_oof, C=C, hidden=hidden,
                    n_iter=n_iter, lr=lr, seed=seed)
    _add(np.clip(fn(X_te_s), 0, 1), np.clip(fn(X_val_oof), 0, 1),
         {'type': 'mlp', 'hidden': hidden, 'n_iter': n_iter, 'lr': lr, 'C': C, 'seed': seed})
    print(f"    {time.time()-t1:.1f}s  total={time.time()-t0:.1f}s")

# === OOF weight optimization ===
print(f"OOF weight optimization ({len(all_preds)} models)...")
val_stack  = np.array(all_val,   dtype=np.float64)   # (n_m, n_test, n_tar)
test_stack = np.array(all_preds, dtype=np.float64)
n_m = len(all_preds)

# Build full-tar val targets
y_val_full = np.zeros((len(X_val_oof), n_tar), dtype=np.float64)
y_val_full[:, valid_idx] = y_val_oof

# Mean column-wise log loss on val (vectorized)
def oof_loss(log_w):
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    blend = np.einsum('m,mjt->jt', w, val_stack)  # (n_val, n_tar)
    blend = np.clip(blend, 1e-7, 1 - 1e-7)
    ll = -(y_val_full * np.log(blend) + (1 - y_val_full) * np.log(1 - blend))
    return ll.mean()

w0   = np.zeros(n_m)
t_opt = time.time()
res  = scipy_minimize(oof_loss, w0, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-9})
opt_w = np.exp(res.x - res.x.max()); opt_w /= opt_w.sum()
print(f"  OOF val loss: {res.fun:.6f}  ({time.time()-t_opt:.1f}s)")
print(f"  Weights: min={opt_w.min():.4f}  max={opt_w.max():.4f}")
all_weights = opt_w.tolist()

# === Two-pass: retrain top models on full training data ===
# Using X_tr_s (100% data) vs X_tr_oof (95%) gives ~5% more samples,
# improving test predictions while keeping OOF-optimized weights.
if tr() > 20:
    _top_k = min(25, n_m)
    _top_idx = np.argsort(opt_w)[-_top_k:][::-1]  # highest weight first
    _retrained = 0
    print(f"Two-pass: retraining up to {_top_k} top models on full data...")
    for mi in _top_idx:
        if tr() < 12:
            break
        cfg = model_cfgs[mi]
        if cfg is None:
            continue  # nystroem / modality — skip
        t1 = time.time()
        if cfg['type'] == 'ridge':
            r2 = Ridge(alpha=cfg['alpha'], fit_intercept=True)
            r2.fit(X_tr_s, ytr_v)
            new_p = full_pred(np.clip(r2.predict(X_te_s), 0, 1)).astype(np.float64)
        elif cfg['type'] == 'logreg':
            fn2 = vec_logreg(X_tr_s, ytr_v, C=cfg['C'], n_iter=cfg['n_iter'], lr=cfg['lr'])
            new_p = full_pred(np.clip(fn2(X_te_s), 0, 1)).astype(np.float64)
        elif cfg['type'] == 'mlp':
            fn2 = vec_2layer(X_tr_s, ytr_v, C=cfg['C'], hidden=cfg['hidden'],
                             n_iter=cfg['n_iter'], lr=cfg['lr'], seed=cfg['seed'])
            new_p = full_pred(np.clip(fn2(X_te_s), 0, 1)).astype(np.float64)
        else:
            continue
        test_stack[mi] = new_p
        _retrained += 1
        print(f"  Retrained #{_retrained} (idx={mi}, {cfg['type']}, w={opt_w[mi]:.4f}): "
              f"{time.time()-t1:.1f}s  t={time.time()-t0:.1f}s")
    print(f"  Two-pass done: {_retrained} models retrained on full data")

# === Weighted ensemble (OOF-optimized weights) ===
print(f"Ensembling {len(all_preds)} models...")
w = np.array(all_weights, dtype=np.float32)
w /= w.sum()
ensemble = np.einsum('m,mjt->jt', w.astype(np.float64), test_stack).astype(np.float32)
ensemble = np.clip(ensemble, 1e-6, 1 - 1e-6)

# === Per-target Platt scaling + Bayesian fallback ===
# For targets with ≥8 positives in OOF val: fit σ,δ on val predictions
# via L-BFGS-B to directly optimize log loss per target.
# Rare targets (< 8 positives in val) fall back to Bayesian shrinkage.
print("Calibrating (Platt + Bayesian fallback)...")
val_ens = np.einsum('m,mjt->jt', opt_w.astype(np.float64), val_stack).astype(np.float64)
val_ens = np.clip(val_ens, 1e-6, 1 - 1e-6)
val_logit = np.log(val_ens / (1 - val_ens))  # (n_val, n_tar)
ens_logit = np.log(np.clip(ensemble, 1e-6, 1-1e-6).astype(np.float64) /
                   (1 - np.clip(ensemble, 1e-6, 1-1e-6).astype(np.float64)))

PLATT_MIN_POS = 8
_platt_n = 0
for j in range(n_tar):
    br = float(base_rates[j])
    n_pos = int(y_val_full[:, j].sum())
    if n_pos >= PLATT_MIN_POS:
        lp = val_logit[:, j]
        yj = y_val_full[:, j]
        def _obj(p, _lp=lp, _y=yj):
            s, d = p
            q = 1 / (1 + np.exp(-np.clip(s * _lp + d, -50, 50)))
            return -((_y * np.log(q+1e-9) + (1-_y) * np.log(1-q+1e-9))).mean() \
                   + 0.05 * (s-1)**2 + 0.05 * d**2
        rj = scipy_minimize(_obj, [1.0, 0.0], method='L-BFGS-B',
                            options={'maxiter': 100})
        s_j, d_j = rj.x
        ensemble[:, j] = np.clip(
            1 / (1 + np.exp(-np.clip(s_j * ens_logit[:, j] + d_j, -50, 50))),
            1e-6, 1-1e-6)
        _platt_n += 1
    else:
        alpha = 0.10 if br < 0.001 else 0.07 if br < 0.003 else \
                0.05 if br < 0.007 else 0.03 if br < 0.02 else 0.01
        ensemble[:, j] = (1.0 - alpha) * ensemble[:, j] + alpha * br
print(f"  Platt: {_platt_n}/{n_tar}  Bayesian: {n_tar-_platt_n}/{n_tar}")

ensemble = np.clip(ensemble.astype(np.float32), 1e-6, 1 - 1e-6)

# === Build submission ===
submission = pd.DataFrame(
    np.full((len(test_features), n_tar), 0.001, dtype=np.float32),
    columns=target_cols,
)
submission.insert(0, "sig_id", test_features["sig_id"].values)
trt_idx = np.where(~ctrl_te)[0]
submission.iloc[trt_idx, 1:] = ensemble

submission.to_csv("submission.csv", index=False)
print(f"\nDone! t={time.time()-t0:.1f}s")
print(f"Submission: {len(submission)} rows x {n_tar} targets")
