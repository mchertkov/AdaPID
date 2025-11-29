# ===== API CELL 1 (verbatim) =====
# =========================
# Imports, config, helpers
# =========================
import math, os, sys
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
from dataclasses import dataclass

np.set_printoptions(suppress=True, linewidth=140)
os.makedirs("figs", exist_ok=True)

def set_seed(seed=0):
    rng = np.random.default_rng(seed)
    return rng

def display_head(arr, n=3, name="arr"):
    print(f"{name}.shape = {arr.shape}")
    print(arr[:n])


# ===== API CELL 2 (verbatim) =====
# ===================================
# Target: Gaussian Mixture 
# ===================================

@dataclass
class GMM:
    MU: np.ndarray    # (K,d)
    SIG: np.ndarray   # (K,) isotropic std per component
    PI: np.ndarray    # (K,) weights, sum to 1

    @property
    def K(self): return int(self.MU.shape[0])
    @property
    def d(self): return int(self.MU.shape[1])

    def describe(self, name="GMM"):
        K, d = self.K, self.d
        s = np.array2string(self.SIG, precision=3)
        p = np.array2string(self.PI, precision=3)
        print(f"{name}: K={K}, d={d}, σ={s[:20]+'...' if len(s)>24 else s}, weights={p[:20]+'...' if len(p)>24 else p}")

def make_grid_3x3(a=1.5, sigma=0.3):
    xs = np.array([-a, 0.0, a], float)
    MU = np.array([[xi, yi] for yi in xs for xi in xs], float)  # row-major
    SIG = np.full(9, float(sigma))
    PI  = np.full(9, 1.0/9.0)
    return GMM(MU=MU, SIG=SIG, PI=PI)

def sample_gmm(gmm: GMM, n: int, rng=None):
    rng = rng or np.random.default_rng()
    K, d = gmm.K, gmm.d
    idx = rng.choice(K, size=n, p=gmm.PI)
    Z = rng.normal(size=(n, d))
    X = gmm.MU[idx] + (gmm.SIG[idx])[:, None] * Z
    return X, idx

def logpdf_gmm(gmm: GMM, X: np.ndarray):
    # isotropic, per-component
    K, d = gmm.K, gmm.d
    X = np.asarray(X, float)
    dif = X[:, None, :] - gmm.MU[None, :, :]        # (n,K,d)
    r2  = np.sum(dif**2, axis=2)                    # (n,K)
    sig2 = gmm.SIG**2                                # (K,)
    logdet = d*np.log(gmm.SIG) + d*np.log(np.sqrt(2*np.pi))
    comp = -0.5*(r2 / sig2[None, :]) - logdet[None, :]
    # log-sum-exp over components with weights
    logw = np.log(gmm.PI)[None, :]
    a = np.max(comp + logw, axis=1, keepdims=True)
    lse = a + np.log(np.sum(np.exp(comp + logw - a), axis=1, keepdims=True))
    return lse.ravel()


# ===== API CELL 3 (verbatim) =====
# --- GMM.sample shim (add right after your Target GMM API cell) ---
import numpy as np

def _gmm_sample_generic(gmm, n, seed=None):
    """
    Draw n iid samples from a Gaussian Mixture gmm with fields:
      gmm.MU:  (K,d) means
      gmm.SIG: scalar, (K,), (d,), (K,d), (d,d), or (K,d,d) covariance/std encodings
               - scalar: shared isotropic σ
               - (K,): per-component isotropic σ_k
               - (d,): shared diagonal std per dim
               - (K,d): per-component diagonal std per dim
               - (d,d): shared full covariance
               - (K,d,d): per-component full covariance
      gmm.PI:  (K,) nonnegative weights (will be normalized)
    Returns: (n,d) array
    """
    rng = np.random.default_rng(seed)
    MU  = np.asarray(gmm.MU, float)           # (K,d)
    PI  = np.asarray(gmm.PI, float).copy()
    PI  = PI / PI.sum()
    SIG = np.asarray(gmm.SIG, float)
    K, d = MU.shape

    # choose components
    idx = rng.choice(K, size=int(n), p=PI)
    means = MU[idx]  # (n,d)

    # sampling helpers
    if SIG.ndim == 0:  # scalar
        std = float(SIG)
        return means + rng.normal(size=(n, d)) * std

    if SIG.ndim == 1:
        if SIG.shape[0] == K:  # per-component isotropic
            stds = SIG[idx][:, None]  # (n,1)
            return means + rng.normal(size=(n, d)) * stds
        if SIG.shape[0] == d:  # shared diagonal
            stds = SIG[None, :]  # (1,d)
            return means + rng.normal(size=(n, d)) * stds
        # fallback: treat as scalar
        std = float(SIG.reshape(()))
        return means + rng.normal(size=(n, d)) * std

    if SIG.ndim == 2:
        if SIG.shape == (K, d):  # per-component diagonal stds
            stds = SIG[idx]  # (n,d)
            return means + rng.normal(size=(n, d)) * stds
        if SIG.shape == (d, d):  # shared full covariance
            L = np.linalg.cholesky(SIG)
            Z = rng.normal(size=(n, d))
            return means + Z @ L.T

    if SIG.ndim == 3 and SIG.shape[0] == K and SIG.shape[1] == SIG.shape[2] == d:
        # per-component full covariance
        X = np.empty((n, d), float)
        for k in range(K):
            mask = (idx == k)
            m = int(mask.sum())
            if m == 0:
                continue
            L = np.linalg.cholesky(SIG[k])
            Z = rng.normal(size=(m, d))
            X[mask] = MU[k] + Z @ L.T
        return X

    raise ValueError(f"Unrecognized SIG shape {SIG.shape} for GMM sampling.")

# Monkey-patch only if missing
if 'GMM' in globals() and not hasattr(GMM, "sample"):
    def _sample_method(self, n, seed=None):
        return _gmm_sample_generic(self, n, seed)
    GMM.sample = _sample_method
    print("[GMM] Added .sample(n, seed) method.")
else:
    print("[GMM] .sample already present (no changes).")


# ===== API CELL 4 (verbatim) =====
# =====================================================
# Schedule API — Single Source of Truth
#   • Constant-β: exact closed forms
#   • PWC-β: exact per-piece formulas
#   • alpha_K_gamma_from_schedule: one canonical helper
# =====================================================

import math, numpy as np

# ---------- Constant-β closed forms ----------
def _a_minus_const(t: float, beta: float) -> float:
    r = math.sqrt(float(beta)); dt = max(0.0, 1.0 - float(t))
    if r == 0.0:
        return float('inf') if dt == 0.0 else 1.0/dt
    s = math.sinh(r*dt)
    return float('inf') if s == 0.0 else r * (math.cosh(r*dt)/s)   # r*coth(r(1-t))

def _b_minus_const(t: float, beta: float) -> float:
    r = math.sqrt(float(beta)); dt = max(0.0, 1.0 - float(t))
    if r == 0.0:
        return float('inf') if dt == 0.0 else 1.0/dt
    s = math.sinh(r*dt)
    return float('inf') if s == 0.0 else r / s                     # r/sinh(r(1-t))

def _c_minus_const(t: float, beta: float) -> float:
    return _a_minus_const(t, beta)                                 # in const-β, c^- = a^-

def _a_plus_1_const(beta: float) -> float:
    r = math.sqrt(float(beta))
    if r == 0.0:
        return 1.0
    s = math.sinh(r)
    return float('inf') if s == 0.0 else r * (math.cosh(r)/s)      # r*coth r

class BetaScheduleConst:
    """Constant-β schedule via exact closed forms."""
    def __init__(self, beta: float):
        self.beta = float(beta)
        self._a_plus_1 = _a_plus_1_const(self.beta)
    def a_plus_1(self) -> float:       return self._a_plus_1
    def c_minus(self, t: float) -> float: return _c_minus_const(t, self.beta)
    def a_minus(self, t: float) -> float: return _a_minus_const(t, self.beta)
    def b_minus(self, t: float) -> float: return _b_minus_const(t, self.beta)

# ---------- PWC-β exact per §2.1.2 (NO shim equalities) ----------
class BetaSchedulePWC:
    @staticmethod
    def constant(beta: float):
        # IMPORTANT: constant-β must be the closed-form schedule
        return BetaScheduleConst(float(beta))

    @staticmethod
    def from_segments(betas, splits):
        return BetaSchedulePWC(betas, splits)

    def __init__(self, betas, splits):
        betas  = np.asarray(betas,  float)
        splits = np.asarray(splits, float)
        assert splits[0] == 0.0 and splits[-1] == 1.0 and np.all(np.diff(splits) > 0)
        assert betas.size + 1 == splits.size
        self.betas, self.splits = betas, splits
        m = betas.size

        # Rightmost piece uses constant-β formulas
        def last_piece(beta_i):
            def a_m(t): return _a_minus_const(t, beta_i)
            def b_m(t): return _b_minus_const(t, beta_i)
            def c_m(t): return _c_minus_const(t, beta_i)
            return a_m, b_m, c_m

        self._pieces = [None]*m
        iR = m-1
        aR, bR, cR = last_piece(betas[iR])
        self._pieces[iR] = (aR, bR, cR)

        # boundary values at the left edge of the rightmost piece
        aR0, bR0, cR0 = aR(splits[iR]), bR(splits[iR]), cR(splits[iR])

        # Backward propagation to earlier pieces (exact §2.1.2)
        for j in reversed(range(m-1)):
            beta_j = float(betas[j]); rj = math.sqrt(beta_j)
            sLj, sRj = float(splits[j]), float(splits[j+1])
            aRj, bRj, cRj = aR0, bR0, cR0

            def make_piece(aRj, bRj, cRj, rj, beta_j, sRj):
                def a_m(t):
                    tau = max(0.0, sRj - float(t))
                    if rj == 0.0:  # β=0 on this piece
                        return aRj
                    th = math.tanh(rj * tau)
                    return rj * (aRj + rj * th) / (rj + aRj * th)
                def b_m(t):
                    at = a_m(t)
                    num = max(0.0, at*at - beta_j)
                    den = max(0.0, aRj*aRj - beta_j)
                    if den == 0.0: return bRj
                    return bRj * math.sqrt(num/den)
                def c_m(t):
                    at = a_m(t)
                    denom = (beta_j - aRj*aRj)
                    if denom == 0.0: return cRj
                    return cRj + (bRj*bRj)/denom * (aRj - at)
                return a_m, b_m, c_m

            aF, bF, cF = make_piece(aRj, bRj, cRj, rj, beta_j, sRj)
            self._pieces[j] = (aF, bF, cF)
            aR0, bR0, cR0 = aF(sLj), bF(sLj), cF(sLj)

        # Forward propagation for a^+(1)
        ap = None
        for k in range(m):
            r  = math.sqrt(float(betas[k]))
            dur = float(splits[k+1] - splits[k])
            if ap is None:
                if r == 0.0: ap = 1.0 if dur >= 1.0 else (float('inf') if dur == 0.0 else 1.0/dur)
                else:
                    s = math.sinh(r*dur); c = math.cosh(r*dur)
                    ap = float('inf') if s == 0.0 else r * (c/s)
            else:
                if r == 0.0:
                    ap = ap
                else:
                    rho = math.exp(-2.0*r*dur) * (ap - r)/(ap + r)
                    ap  = r * (1.0 + rho)/(1.0 - rho)
        self._a_plus_1 = float(ap)

    def _seg_idx(self, t: float) -> int:
        t = float(t)
        if t >= 1.0: return self.betas.size - 1
        i = int(np.searchsorted(self.splits, t, side="right") - 1)
        return max(0, min(i, self.betas.size - 1))

    def a_minus(self, t: float) -> float:
        i = self._seg_idx(t); return float(self._pieces[i][0](t))
    def b_minus(self, t: float) -> float:
        i = self._seg_idx(t); return float(self._pieces[i][1](t))
    def c_minus(self, t: float) -> float:
        i = self._seg_idx(t); return float(self._pieces[i][2](t))
    def a_plus_1(self) -> float:
        return self._a_plus_1

# ---------- One canonical (α, K, γ) ----------
def alpha_K_gamma_from_schedule(sched, t: float):
    """
    K(t) = c^-(t) - a^+(1),  α(t) = b^-(t)/K(t),  γ(t)=1/sqrt(K(t))
    (γ is for diagnostics; the simulator below uses unit diffusion unless you ask otherwise.)
    """
    t = float(t)
    c_m = float(sched.c_minus(t))
    ap1 = float(sched.a_plus_1())
    Kt  = c_m - ap1
    if not (Kt > 0.0 and np.isfinite(Kt)):
        raise RuntimeError(f"K(t) non-positive or non-finite at t={t:.6g}: {Kt}")
    alpha = float(sched.b_minus(t)) / Kt
    gamma = 1.0 / math.sqrt(Kt)
    return alpha, Kt, gamma


# ===== API CELL 5 (verbatim) =====
# =====================================================
# Oracle ŷ(X,t) for isotropic GMM + optimal control u*
# =====================================================

def yhat_oracle_gmm(X, t, sched, gmm):
    X = np.asarray(X, float)                # (M,d)
    alpha, Kt, _ = alpha_K_gamma_from_schedule(sched, t)
    m = alpha * X                           # (M,d)

    inv_sig2 = 1.0 / (gmm.SIG**2)           # (K,)
    lam = (Kt * inv_sig2) / (inv_sig2 + Kt) # (K,)

    dmu = gmm.MU[None, :, :] - m[:, None, :]   # (M,K,d)
    r2  = np.sum(dmu**2, axis=2)               # (M,K)
    wexp = np.exp(-0.5 * r2 * lam[None, :]) * gmm.PI[None, :]  # (M,K)

    num = inv_sig2[None, :, None] * gmm.MU[None, :, :] + Kt * m[:, None, :]
    den = (inv_sig2 + Kt)[None, :, None]
    mu_t = num / den                           # (M,K,d)

    Z = np.sum(wexp, axis=1, keepdims=True) + 1e-300
    return np.sum(wexp[:, :, None] * mu_t, axis=1) / Z          # (M,d)

def control_u_star(X, t, sched, gmm):
    yhat = yhat_oracle_gmm(X, t, sched, gmm)
    a_m  = float(sched.a_minus(t))
    b_m  = float(sched.b_minus(t))
    return b_m * yhat - a_m * X


# ===== API CELL 6 (verbatim) =====
# =====================================================
# Simulator (Euler–Maruyama, unit diffusion) + varsigma
#   • Interior time grid (avoid t=0 and t=1)
#   • API unchanged
# =====================================================

import numpy as np, math

def time_grid(T: int, scheme: str = "mid"):
    if scheme == "mid":
        return (np.arange(T) + 0.5) / T
    elif scheme == "open-right":
        return (np.arange(T) + 1.0) / (T + 1.0)
    elif scheme == "right":
        return np.linspace(1.0/T, 1.0, T, endpoint=True)         # touches 1 (avoid if possible)
    elif scheme == "left":
        return np.linspace(0.0, 1.0 - 1.0/T, T, endpoint=True)   # touches 0 (avoid)
    else:
        raise ValueError("scheme must be 'mid', 'open-right', 'right', or 'left'")

def simulate_paths_oracle(sched, gmm, M: int, T: int, seed: int = 0,
                          return_budgets: bool = False,
                          drift_scheme: str = "mid"):
    rng = np.random.default_rng(seed)
    dt  = 1.0 / T
    ts  = time_grid(T, drift_scheme)

    d = gmm.d
    X = rng.normal(size=(M, d))     # same as your experiment setup

    budgets = [] if return_budgets else None
    varsigma = 0.0

    for t_eval in ts:
        t_eval = float(t_eval)
        u = control_u_star(X, t_eval, sched, gmm)
        dW = rng.normal(size=X.shape) * math.sqrt(dt)            # unit diffusion (per your design)
        X  = X + u * dt + dW
        if return_budgets:
            nu2_avg = float(np.sum(u*u) / M)
            varsigma += nu2_avg * dt
            budgets.append({"t": t_eval, "norm_u_avg": float(np.sqrt(nu2_avg)), "varsigma": float(varsigma)})
    return (X, budgets) if return_budgets else X


# ===== API CELL 7 (verbatim) =====
# ==========================================================
# Empirical W2 between equally-sized samples (balanced).
# Hungarian (SciPy) if available; else a small Sinkhorn.
# ==========================================================
def _hungarian_or_none(cost):
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:
        return None
    r, c = linear_sum_assignment(cost)
    return r, c

def w2_empirical(X, Y, prefer_hungarian=True, sinkhorn_eps=0.01, sinkhorn_iters=300):
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    assert X.shape == Y.shape and X.ndim == 2
    n, d = X.shape
    C = np.sum((X[:, None, :] - Y[None, :, :])**2, axis=2)  # (n,n)

    if prefer_hungarian:
        rc = _hungarian_or_none(C)
        if rc is not None:
            r, c = rc
            return float(np.mean(C[r, c])), "hungarian"

    # tiny Sinkhorn on uniform marginals
    K = np.exp(-C / max(sinkhorn_eps, 1e-6))
    u = np.ones(n) / n
    v = np.ones(n) / n
    a = np.ones(n); b = np.ones(n)
    for _ in range(sinkhorn_iters):
        a = u / (K @ b + 1e-300)
        b = v / (K.T @ a + 1e-300)
    P = (a[:, None] * K) * b[None, :]
    w2 = float(np.sum(P * C))
    return w2, "sinkhorn"


# ===== API CELL 8 (verbatim) =====
# --------- utilities: simulation to terminal, norms, CRN sampling ----------
def simulate_terminal_const_beta(beta, gmm, *, M=3000, T=1200, seed=0, t_floor=0.01):
    rng  = np.random.default_rng(seed)
    dt   = 1.0 / T
    d    = gmm.d
    X    = np.zeros((M, d), float)   # start at 0 for stability/consistency

    sched = BetaSchedulePWC.constant(float(beta))
    _ = alpha_K_gamma_from_schedule(sched, 0.5)   # interior health probe

    for n in range(T):
        t_mid = (n + 0.5)/T
        t_eff = min(max(t_mid, float(t_floor)), 1.0 - float(t_floor))

        yhat = yhat_oracle_gmm(X, float(t_eff), sched, gmm)
        a_m  = float(sched.a_minus(float(t_eff)))
        b_m  = float(sched.b_minus(float(t_eff)))
        u    = b_m * yhat - a_m * X

        dW = rng.normal(size=X.shape) * math.sqrt(dt)
        X  = X + u*dt + dW

    return X

def euclid2(X):
    return np.sum(X*X, axis=1)

# --------- OT core: exact (Hungarian) if available; else Sinkhorn ----------
def _wasserstein2_exact(X, Y):
    """Exact W2 via Hungarian if SciPy is available; else raise."""
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:
        raise RuntimeError("SciPy not available for exact OT; use method='sinkhorn'.")
    C = np.sum((X[:,None,:] - Y[None,:,:])**2, axis=2)  # (K,K)
    r,c = linear_sum_assignment(C)
    return float(np.mean(C[r,c]))

def _sinkhorn_w2(X, Y, reg=0.02, n_iter=400, tol=1e-6):
    """
    Entropic OT (uniform marginals) returning <C, P*> ; for quadratic cost this
    is an upper bound on W2^2 as reg>0. Good enough for comparisons.
    """
    K = X.shape[0]
    C = np.sum((X[:,None,:] - Y[None,:,:])**2, axis=2)      # (K,K)
    Kmat = np.exp(-C / max(reg,1e-8)) + 1e-300              # Gibbs kernel
    u = np.ones(K)/K; v = np.ones(K)/K
    a = np.ones(K)/K; b = np.ones(K)/K

    for _ in range(n_iter):
        u_prev = u
        u = a / (Kmat @ v + 1e-300)
        v = b / (Kmat.T @ u + 1e-300)
        if np.max(np.abs(u - u_prev)) < tol:
            break
    P = (u[:,None] * Kmat) * v[None,:]                      # coupling
    return float(np.sum(P * C))

def wasserstein2_empirical(X, Y, method="auto", reg=0.02):
    """
    Returns W2^2 between two equally sized sets (uniform weights).
    method: 'auto' (exact if possible else sinkhorn), 'exact', or 'sinkhorn'
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    assert X.shape == Y.shape and X.ndim == 2
    if method == "exact":
        return _wasserstein2_exact(X, Y)
    if method == "sinkhorn":
        return _sinkhorn_w2(X, Y, reg=reg)
    # auto
    try:
        return _wasserstein2_exact(X, Y)
    except Exception:
        return _sinkhorn_w2(X, Y, reg=reg)
