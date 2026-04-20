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
    svg  = TruncatedSVD(80, n_iter=1, random_state=0).fit_transform(ag)
    svc  = TruncatedSVD(40, n_iter=1, random_state=0).fit_transform(ac)
    svgc = TruncatedSVD(30, n_iter=1, random_state=0).fit_transform(agc)
    # Cross-PCA: SVD on gene×cell outer products captures interaction manifold
    n_all = len(all_g)
    cross_flat = (svg[:, :20, None] * svc[:, None, :15]).reshape(n_all, -1)
    cross_pca = TruncatedSVD(20, n_iter=1, random_state=0).fit_transform(cross_flat)
    cp_t = np.concatenate([train_df["cp_t"].values, test_df["cp_t"].values])
    cp_d = np.concatenate([train_df["cp_d"].values, test_df["cp_d"].values])
    X = np.hstack([
        svg, svc, svgc,
        svg[:, :20] * cp_t[:, None], svg[:, :20] * cp_d[:, None],
        svc[:, :10] * cp_t[:, None], svc[:, :10] * cp_d[:, None],
        svg[:, :8] * svc[:, :8],
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


all_preds   = []
all_weights = []

# Budget: leave 27s for ensemble/write/python-exit
BUDGET = 273.0
tr = lambda: BUDGET - (time.time() - t0)

# --- Ridge (multi-output, closed-form) ---
print("Ridge...")
for alpha in [30.0, 100.0, 500.0]:
    t1 = time.time()
    r = Ridge(alpha=alpha, fit_intercept=True)
    r.fit(X_tr_s, ytr_v)
    pv = np.clip(r.predict(X_te_s), 0.0, 1.0).astype(np.float32)
    all_preds.append(full_pred(pv)); all_weights.append(0.3)
    print(f"  alpha={alpha}: {time.time()-t1:.1f}s")

# --- Vectorized LogReg ensemble (warm-init, 60 iters) ---
print("LogReg ensemble...")
for C in [0.03, 0.07, 0.15, 0.30, 0.60]:
    if tr() < 8:
        print(f"  Skipping C={C}, {tr():.0f}s remain")
        break
    t1 = time.time()
    fn = vec_logreg(X_tr_s, ytr_v, C=C, n_iter=60, lr=0.01)
    pv = fn(X_te_s)
    all_preds.append(full_pred(pv)); all_weights.append(0.6)
    print(f"  C={C}: {time.time()-t1:.1f}s  total={time.time()-t0:.1f}s")

# --- Nystroem RBF kernel + LogReg ---
# 150 components: ~7s transform + 3×9s LR = 34s total
if tr() > 90:
    print("Nystroem + LogReg...")
    t1 = time.time()
    ny = Nystroem(kernel='rbf', n_components=150, random_state=0)
    X_tr_ny = ny.fit_transform(X_tr_s).astype(np.float32)
    X_te_ny = ny.transform(X_te_s).astype(np.float32)
    sc_ny = StandardScaler()
    X_tr_ny = sc_ny.fit_transform(X_tr_ny).astype(np.float32)
    X_te_ny = sc_ny.transform(X_te_ny).astype(np.float32)
    print(f"  Nystroem transform: {time.time()-t1:.1f}s")
    for C in [0.1, 0.3, 0.7]:
        if tr() < 12:
            break
        t1 = time.time()
        fn = vec_logreg(X_tr_ny, ytr_v, C=C, n_iter=40, lr=0.01)
        pv = fn(X_te_ny)
        all_preds.append(full_pred(pv)); all_weights.append(0.8)
        print(f"  Ny+LR C={C}: {time.time()-t1:.1f}s  total={time.time()-t0:.1f}s")
else:
    print(f"Skipping Nystroem, {tr():.0f}s remain")

# --- Vectorized 2-layer MLP ensemble ---
# h=64 60iter ≈ 18s; h=128 50iter ≈ 15-32s (variable).
# Use continue to skip h=128 when budget is tight but keep running h=64 models.
print("2-layer MLP ensemble...")
mlp2_configs = [
    (0,  64,  60, 0.005, 0.10),
    (0,  128, 50, 0.003, 0.10),
    (0,  256, 35, 0.002, 0.10),
    (1,  64,  60, 0.005, 0.10),
    (1,  128, 50, 0.003, 0.10),
    (1,  256, 35, 0.002, 0.10),
    (2,  64,  60, 0.005, 0.10),
    (3,  64,  60, 0.005, 0.10),
    (0,  64,  60, 0.005, 0.20),
    (0,  128, 50, 0.003, 0.20),
    (4,  64,  60, 0.005, 0.10),
    (1,  64,  60, 0.005, 0.20),
    (5,  64,  60, 0.005, 0.10),
    (2,  64,  60, 0.005, 0.20),
]
for seed, hidden, n_iter, lr, C in mlp2_configs:
    remaining = tr()
    if remaining < 23:
        print(f"  Stopping, {remaining:.0f}s remain")
        break
    if hidden >= 256 and remaining < 65:
        print(f"  Skipping h={hidden} s={seed} C={C}, {remaining:.0f}s remain")
        continue
    if hidden >= 128 and remaining < 42:
        print(f"  Skipping h={hidden} s={seed} C={C}, {remaining:.0f}s remain")
        continue
    t1 = time.time()
    print(f"  2L h={hidden} s={seed} C={C}  t={time.time()-t0:.1f}s ...")
    fn = vec_2layer(X_tr_s, ytr_v, C=C, hidden=hidden,
                    n_iter=n_iter, lr=lr, seed=seed)
    pv = np.clip(fn(X_te_s), 0.0, 1.0)
    all_preds.append(full_pred(pv)); all_weights.append(1.0)
    print(f"    {time.time()-t1:.1f}s  total={time.time()-t0:.1f}s")

# === Weighted ensemble ===
print(f"Ensembling {len(all_preds)} models...")
w = np.array(all_weights, dtype=np.float32)
w /= w.sum()
ensemble = sum(float(wi) * pi for wi, pi in zip(w, all_preds)).astype(np.float32)
ensemble = np.clip(ensemble, 1e-6, 1 - 1e-6)

# === Per-target adaptive Bayesian calibration ===
print("Calibrating...")
for j in range(n_tar):
    br = float(base_rates[j])
    alpha = 0.10 if br < 0.001 else 0.07 if br < 0.003 else \
            0.05 if br < 0.007 else 0.03 if br < 0.02 else 0.01
    ensemble[:, j] = (1.0 - alpha) * ensemble[:, j] + alpha * br

ensemble = np.clip(ensemble, 1e-6, 1 - 1e-6)

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
