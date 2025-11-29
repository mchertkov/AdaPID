import numpy as np
from .api import GMM

# ===== Models (verbatim-style minimal) =====

def build_regular_grid_gmm(a=1.5, sigma=0.3):
    xs = np.array([-a, 0.0, a], float)
    MU  = np.array([[xi, yj] for yj in xs for xi in xs], float)
    SIG = np.full(9, float(sigma))
    PI  = np.ones(9, float)/9.0
    return GMM(MU, SIG, PI)

def build_perturbed_gmm(seed, a=1.5, base_sigma=0.3,
                        jitter_frac=1/3, sig_scale=(0.5, 2.0),
                        weight_range=(0.5, 1.5)):
    rng = np.random.default_rng(int(seed))
    xs = np.array([-a, 0.0, a], float)
    centers = np.array([[xi, yj] for yj in xs for xi in xs], float)
    jitter_max = a * float(jitter_frac)
    MU  = centers + rng.uniform(-jitter_max, jitter_max, size=centers.shape)
    SIG = base_sigma * rng.uniform(sig_scale[0], sig_scale[1], size=9)
    w   = rng.uniform(weight_range[0], weight_range[1], size=9)
    PI  = w / np.sum(w)
    return GMM(MU, SIG, PI)

# Default instances (can be overridden by user code)
regular = build_regular_grid_gmm(a=1.5, sigma=0.3)
pertA   = build_perturbed_gmm(seed=20241005, a=1.5, base_sigma=0.3)
pertB   = build_perturbed_gmm(seed=20241006, a=1.5, base_sigma=0.3)

models  = [("Regular 3×3", regular), ("Perturbed A", pertA), ("Perturbed B", pertB)]
