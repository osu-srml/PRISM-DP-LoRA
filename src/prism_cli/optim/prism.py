"""prism.py

PRISM: Projected Riemannian Invariant Subspace Mechanism (for DP-LoRA)

This module implements a *gauge-invariant* DP training mechanism for LoRA by
operating on the intrinsic rank-r update Z = A B^T instead of directly noising
(A, B) in a gauge-dependent way.

Key ideas (paper-aligned):
  1) Intrinsic tangent projection: P(G) = G - P_A^\perp G P_B^\perp.
  2) DP clipping/noise in the tangent space (global across LoRA modules).
  3) Retraction back to rank r without the bilinear DP noise term.

What PRISM adds on top of the base "GILA/LoRAI" mechanism:
  - Spectral initialization helper (principal SVD init that accelerates convergence).
  - Balanced tiny full-rank LoRA initialization for no-Spectral ablations.
  - Noise-immune adaptive update (DP-Aware Riemannian Adam):
      * DP-aware eigenvalue floor + condition-number clamp
      * optional floor modes for implementation ablations
      * optional one-step delayed rank-space preconditioner
      * optional horizontal/min-norm lift gauge fixing before moments/retraction
      * optional first-moment horizontalization relative to the current tangent space
      * optional post-preconditioner update clipping
  - Rich DP logging hooks (SNR, clip stats, preconditioner spectrum).

Designed to be dropped into the existing repo without touching other files.

NOTE
- This code assumes PEFT LoRA modules follow HuggingFace/peft conventions.
- DP accounting is handled externally (e.g., Opacus PrivacyEngine). PRISM uses
  Opacus only for per-sample gradients (grad_sample).
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor


# -----------------------------------------------------------------------------
# Shape utilities (PEFT LoRA + Opacus grad_sample)
# -----------------------------------------------------------------------------


class _LoraShapeHelper:
    """Consistent reshaping for LoRA params and Opacus per-sample gradients.

    PEFT LoRA mapping (Linear LoRA):
      - lora_A.weight: (r, in_features)
      - lora_B.weight: (out_features, r)

    We define:
      - B := lora_A.weight^T  (in_features, r)
      - A := lora_B.weight    (out_features, r)
    so the LoRA update matrix is Z = A B^T.
    """

    @staticmethod
    def move_lora_dim_to_last(x: Tensor, lora_dim: int) -> Tuple[Tensor, Tuple[int, ...]]:
        x2 = torch.moveaxis(x, lora_dim, -1)
        shape_after_move = tuple(x2.shape)
        x2 = x2.reshape(-1, x2.shape[-1])
        return x2, shape_after_move

    @staticmethod
    def restore_param_shape(x2d: Tensor, ref: Tensor, lora_dim: int) -> Tensor:
        _, shape_after_move = _LoraShapeHelper.move_lora_dim_to_last(ref, lora_dim)
        out = x2d.reshape(shape_after_move)
        out = torch.moveaxis(out, -1, lora_dim)
        return out

    @staticmethod
    def get_grad_tensor(p: torch.nn.Parameter) -> Tensor:
        if p.grad is None:
            return torch.zeros_like(p.data)
        return p.grad.data

    @staticmethod
    def _as_grad_sample_tensor(gs: object) -> Optional[Tensor]:
        if gs is None:
            return None
        if isinstance(gs, Tensor):
            return gs
        if isinstance(gs, (list, tuple)):
            parts = [g for g in gs if isinstance(g, Tensor)]
            if not parts:
                return None
            return torch.cat(parts, dim=0)
        return None

    @staticmethod
    def get_grad_sample_tensor(p: torch.nn.Parameter) -> Optional[Tensor]:
        return _LoraShapeHelper._as_grad_sample_tensor(getattr(p, "grad_sample", None))

    @staticmethod
    def clear_grad_sample(p: torch.nn.Parameter) -> None:
        if hasattr(p, "grad_sample"):
            setattr(p, "grad_sample", None)

    @staticmethod
    def move_lora_dim_to_last_grad_sample(gs_raw: object, lora_dim_in_param: int) -> Tensor:
        """Move LoRA rank dim to last for Opacus per-sample grads.

        Args:
          gs_raw: grad_sample, expected shape (bs, *p.shape)
          lora_dim_in_param: rank dim index in p.shape

        Returns:
          (bs, d, r)
        """
        gs = _LoraShapeHelper._as_grad_sample_tensor(gs_raw)
        if gs is None:
            raise ValueError("grad_sample is None or unsupported type")
        if gs.ndim < 2:
            raise ValueError(f"grad_sample must have >=2 dims, got {gs.ndim}")

        if lora_dim_in_param >= 0:
            lora_dim_in_gs = lora_dim_in_param + 1
        else:
            lora_dim_in_gs = lora_dim_in_param

        gs2 = torch.moveaxis(gs, lora_dim_in_gs, -1)
        bs = gs2.shape[0]
        r = gs2.shape[-1]
        return gs2.reshape(bs, -1, r)


# -----------------------------------------------------------------------------
# Stable PSD linear algebra (r is small; use eig in fp32)
# -----------------------------------------------------------------------------


def _sym(M: Tensor) -> Tensor:
    return 0.5 * (M + M.T)


def _psd_eigh(M: Tensor) -> Tuple[Tensor, Tensor]:
    """Robust eigen decomposition for small PSD matrices.

    torch.linalg.eigh can occasionally fail to converge on ill-conditioned inputs.
    We retry with diagonal jitter and (as a last resort) a CPU+fp64 solve.
    """
    Mf = _sym(M.float())
    try:
        w, v = torch.linalg.eigh(Mf)
    except Exception:
        n = Mf.shape[0]
        # Relative diagonal jitter (scale-aware)
        diag_mean = float(Mf.diagonal().abs().mean().item()) if n > 0 else 0.0
        jitter = (1e-6 * (diag_mean + 1.0))
        J = torch.eye(n, device=Mf.device, dtype=Mf.dtype) * jitter
        try:
            w, v = torch.linalg.eigh(Mf + J)
        except Exception:
            # Last resort: CPU fp64
            w_cpu, v_cpu = torch.linalg.eigh((Mf + J).double().cpu())
            w = w_cpu.to(device=Mf.device, dtype=Mf.dtype)
            v = v_cpu.to(device=Mf.device, dtype=Mf.dtype)
    w = torch.clamp(w, min=0.0)
    return w.to(M.dtype), v.to(M.dtype)


def _psd_pinv(M: Tensor, rcond: float = 1e-6) -> Tensor:
    """PSD pseudoinverse with relative cutoff."""
    w, v = _psd_eigh(M)
    max_w = torch.max(w)
    if float(max_w) <= 0.0:
        return torch.zeros_like(M)

    cutoff = float(rcond) * float(max_w)
    inv = torch.zeros_like(w)
    mask = w > cutoff
    inv[mask] = 1.0 / w[mask]

    # (v * inv) @ v^T
    out = (v * inv.unsqueeze(0)) @ v.T
    return _sym(out)


def _psd_invsqrt_damped(M: Tensor, eps: float = 1e-8) -> Tensor:
    """(M + eps I)^(-1/2) for symmetric PSD M."""
    w, v = _psd_eigh(M)
    w = w + float(eps)
    invsqrt = 1.0 / torch.sqrt(w)
    out = (v * invsqrt.unsqueeze(0)) @ v.T
    return _sym(out)


def _psd_invsqrt_clamped(
    M: Tensor,
    eps: float,
    floor: float,
    cond_max: Optional[float] = None,
    cond_strategy: str = "raise_small",
) -> Tuple[Tensor, Dict[str, float]]:
    """Inverse sqrt with eigenvalue floor and optional condition-number clamp.

    Returns invsqrt(M) and spectrum stats.
    """
    w, v = _psd_eigh(M)
    w = w + float(eps)

    floor_t = torch.tensor(float(floor), device=w.device, dtype=w.dtype)
    w2 = torch.maximum(w, floor_t)

    if cond_max is not None and cond_max > 1.0:
        strategy = str(cond_strategy).lower()
        if strategy in {"raise_small", "floor", "safe"}:
            # Bound the condition number by *raising small eigenvalues* rather
            # than capping large ones. Capping large eigenvalues increases the
            # inverse-square-root in high-energy directions and can amplify
            # noisy updates. This implements: lambda_min >= lambda_max/cond_max.
            max_w = torch.max(w2)
            cond_floor = max_w / float(cond_max)
            w2 = torch.maximum(w2, cond_floor)
        elif strategy in {"cap_large", "legacy", "old"}:
            # Legacy behavior kept only for ablation/reproduction.
            cap = floor_t * float(cond_max)
            w2 = torch.minimum(w2, cap)
        else:
            raise ValueError(f"Unknown cond_strategy={cond_strategy!r}")

    invsqrt = 1.0 / torch.sqrt(w2)
    out = (v * invsqrt.unsqueeze(0)) @ v.T
    stats = {
        "eig_min": float(torch.min(w).item()),
        "eig_max": float(torch.max(w).item()),
        "eig_min_clamped": float(torch.min(w2).item()),
        "eig_max_clamped": float(torch.max(w2).item()),
    }
    return _sym(out), stats


def _apply_projector(A: Tensor, M_pinv: Tensor, X: Tensor) -> Tensor:
    """Apply Π_A = A (A^T A)^+ A^T to X without forming Π_A.

    Supports X: (m,k) or (bs,m,k).
    """
    if X.ndim == 2:
        return A @ (M_pinv @ (A.T @ X))
    if X.ndim == 3:
        AtX = torch.matmul(A.T, X)      # (bs, r, k)
        tmp = torch.matmul(M_pinv, AtX)  # (bs, r, k)
        return torch.matmul(A, tmp)      # (bs, m, k)
    raise ValueError(f"X must be 2D or 3D, got {X.ndim}D")

def _apply_projector_Q(Q: Tensor, X: Tensor) -> Tensor:
    """Apply Π = Q Q^T to X where Q has orthonormal columns.

    Supports X: (m,k) or (bs,m,k).
    """
    if X.ndim == 2:
        return Q @ (Q.T @ X)
    if X.ndim == 3:
        QtX = torch.matmul(Q.T, X)  # (bs, r, k)
        return torch.matmul(Q, QtX)  # (bs, m, k)
    raise ValueError(f"X must be 2D or 3D, got {X.ndim}D")


# -----------------------------------------------------------------------------
# Retraction with gauge alignment (Procrustes)
# -----------------------------------------------------------------------------


def _procrustes_align(A_new: Tensor, B_new: Tensor, A_ref: Tensor, B_ref: Tensor) -> Tuple[Tensor, Tensor]:
    """Right-multiply (A_new, B_new) by an orthogonal Q to align to (A_ref, B_ref)."""
    C = (A_new.float().T @ A_ref.float()) + (B_new.float().T @ B_ref.float())
    U, _, Vh = torch.linalg.svd(C, full_matrices=False)
    Q = (U @ Vh).to(A_new.dtype)
    return A_new @ Q, B_new @ Q


def _retract_rank_r(
    A: Tensor,
    B: Tensor,
    dA: Tensor,
    dB: Tensor,
    eta: float,
    r: int,
    align_to: Optional[Tuple[Tensor, Tensor]] = None,
) -> Tuple[Tensor, Tensor]:
    """Retract Z + eta(dA B^T + A dB^T) back to rank r.

    Implementation uses a rank-(<=2r) representation and SVD of a (2r x 2r) core.

    IMPORTANT: the (eta^2 dA dB^T) bilinear term is intentionally absent.
    """
    if r <= 0:
        raise ValueError("rank r must be positive")

    A0 = A.float()
    B0 = B.float()
    dA0 = dA.float()
    dB0 = dB.float()

    U = torch.cat([A0, dA0], dim=1)  # (m,2r)
    V = torch.cat([B0, dB0], dim=1)  # (n,2r)

    Q_u, R_u = torch.linalg.qr(U, mode="reduced")
    Q_v, R_v = torch.linalg.qr(V, mode="reduced")

    I = torch.eye(r, device=A.device, dtype=torch.float32)
    K = torch.zeros((2 * r, 2 * r), device=A.device, dtype=torch.float32)
    K[:r, :r] = I
    K[:r, r:] = float(eta) * I
    K[r:, :r] = float(eta) * I

    core = R_u @ K @ R_v.T
    Uc, S, Vhc = torch.linalg.svd(core, full_matrices=False)

    rr = min(r, S.numel())
    Uc_r = Uc[:, :rr]
    Vc_r = Vhc.T[:, :rr]
    S_r = S[:rr]

    S_sqrt = torch.diag(torch.sqrt(torch.clamp(S_r, min=0.0)))

    A_new = (Q_u @ Uc_r) @ S_sqrt
    B_new = (Q_v @ Vc_r) @ S_sqrt

    if rr < r:
        A_new = torch.cat([A_new, torch.zeros((A_new.shape[0], r - rr), device=A.device, dtype=A_new.dtype)], dim=1)
        B_new = torch.cat([B_new, torch.zeros((B_new.shape[0], r - rr), device=B.device, dtype=B_new.dtype)], dim=1)

    A_new = A_new.to(dtype=A.dtype)
    B_new = B_new.to(dtype=B.dtype)

    if align_to is not None:
        A_ref, B_ref = align_to
        A_new, B_new = _procrustes_align(A_new, B_new, A_ref, B_ref)

    return A_new, B_new


# -----------------------------------------------------------------------------
# Intrinsic norm helper: ||dZ||_F from (dA,dB,A,B) without forming dZ
# -----------------------------------------------------------------------------


def _tangent_fro_norm_sq(dA: Tensor, dB: Tensor, A: Tensor, B: Tensor) -> Tensor:
    """Compute ||dZ||_F^2 for dZ = dA B^T + A dB^T.

    Works for:
      dA,dB: (m,r),(n,r) -> scalar
      dA,dB: (bs,m,r),(bs,n,r) -> (bs,)

    Uses: ||dZ||^2 = tr(dA^T dA N) + tr(dB^T dB M) + 2 tr((A^T dA)(B^T dB))
    where M=A^T A, N=B^T B.
    """
    M = A.T @ A
    N = B.T @ B

    if dA.ndim == 2:
        dA_t_dA = dA.T @ dA
        dB_t_dB = dB.T @ dB
        At_dA = A.T @ dA
        Bt_dB = B.T @ dB
        term1 = torch.sum(dA_t_dA * N)
        term2 = torch.sum(dB_t_dB * M)
        term3 = torch.sum(At_dA * Bt_dB.T)
        return term1 + term2 + 2.0 * term3

    if dA.ndim == 3:
        dA_t_dA = torch.matmul(dA.transpose(1, 2), dA)  # (bs,r,r)
        dB_t_dB = torch.matmul(dB.transpose(1, 2), dB)
        At_dA = torch.matmul(A.T, dA)
        Bt_dB = torch.matmul(B.T, dB)
        term1 = torch.einsum("bij,ij->b", dA_t_dA, N)
        term2 = torch.einsum("bij,ij->b", dB_t_dB, M)
        term3 = torch.einsum("bij,bji->b", At_dA, Bt_dB)
        return term1 + term2 + 2.0 * term3

    raise ValueError(f"dA must be 2D or 3D, got {dA.ndim}D")


def _z_fro_norm_sq(A: Tensor, B: Tensor) -> Tensor:
    """Compute ||Z||_F^2 for Z = A B^T without forming Z.

    Uses: ||Z||_F^2 = tr((A^T A)(B^T B)).
    """
    M = A.T @ A
    N = B.T @ B
    # N is symmetric; tr(MN) = sum_{ij} M_ij N_ji = sum(M * N)
    return torch.sum(M * N)


def _psd_inv_from_invsqrt(M_invsqrt: Tensor) -> Tensor:
    """Return (M+eps I)^(-1) from a (damped) (M+eps I)^(-1/2)."""
    return _sym(M_invsqrt @ M_invsqrt)


# -----------------------------------------------------------------------------
# Lift gauge fixing: choose a horizontal/min-factor-norm representative
# -----------------------------------------------------------------------------


def _sylvester_symmetric_solve(M: Tensor, N: Tensor, C: Tensor, eps: float = 1e-12) -> Tensor:
    """Solve M X + X N = C for small symmetric PSD M,N.

    The LoRA rank r is small, so an eigendecomposition-based Sylvester solve is
    simple and robust.  The result is used only as deterministic DP-safe
    post-processing to choose a canonical factor lift of an intrinsic tangent.
    """
    Mf = _sym(M.float())
    Nf = _sym(N.float())
    Cf = C.float()

    wm, Um = _psd_eigh(Mf)
    wn, Vn = _psd_eigh(Nf)

    denom = wm[:, None] + wn[None, :]
    denom = torch.clamp(denom, min=float(eps))
    C_tilde = Um.T @ Cf @ Vn
    X_tilde = C_tilde / denom
    X = Um @ X_tilde @ Vn.T
    return X.to(device=C.device, dtype=C.dtype)


def _horizontalize_lift(dA: Tensor, dB: Tensor, A: Tensor, B: Tensor, eps: float = 1e-12) -> Tuple[Tensor, Tensor]:
    """Gauge-fix a tangent lift without changing the intrinsic tangent dZ.

    A tangent lift is non-unique: for any Omega,
        (dA, dB) -> (dA + A Omega, dB - B Omega^T)
    induces the same dZ = dA B^T + A dB^T.  This function chooses the
    minimum-factor-norm / horizontal representative by solving
        (A^T A) Omega + Omega (B^T B) = dB^T B - A^T dA.

    This is deterministic post-processing of the DP-sanitized tangent update.
    It can reduce lift-dependent second-moment noise without changing the
    first-order intrinsic update used by PRISM.
    """
    if dA.ndim != 2 or dB.ndim != 2:
        raise ValueError("_horizontalize_lift currently expects 2D dA,dB")
    M = _sym(A.float().T @ A.float())
    N = _sym(B.float().T @ B.float())
    rhs = dB.float().T @ B.float() - A.float().T @ dA.float()
    Omega = _sylvester_symmetric_solve(M, N, rhs, eps=eps).to(device=dA.device, dtype=dA.dtype)
    return dA + A.to(dtype=dA.dtype) @ Omega, dB - B.to(dtype=dB.dtype) @ Omega.T



# -----------------------------------------------------------------------------
# DP state + logging
# -----------------------------------------------------------------------------


@dataclass
class _DPAccumState:
    max_grad_norm: float
    total_samples: int = 0

    # clipping stats across microbatches
    microbatches: int = 0
    clipped_frac_sum: float = 0.0
    coef_mean_sum: float = 0.0
    coef_min_sum: float = 0.0

    # last microbatch detailed stats (for debugging)
    last_micro_stats: Dict[str, float] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# PRISM Optimizer
# -----------------------------------------------------------------------------


class PRISM(torch.optim.Optimizer):
    """PRISM optimizer implementing intrinsic DP-LoRA with noise-immune adaptivity.

    Usage patterns:

    Non-DP:
      opt = PRISM(lora_params, lr=..., use_adaptive=True)
      loss.backward(); opt.step()

    DP (custom):
      opt.dp_begin(max_grad_norm=C)
      for microbatch:
        loss.backward(); opt.dp_accumulate()
      opt.dp_finalize(noise_multiplier=sigma)

    Notes:
      - DP clipping is **global across all LoRA modules** in the concatenated
        intrinsic tangent Frobenius norm.
      - Any additional deterministic transforms (alignment, update clipping,
        eigenvalue clamping) are DP-safe as post-processing.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        rcond: float = 1e-6,
        lora_l_dim: int = 0,
        lora_r_dim: int = -1,
        use_adaptive: bool = True,
        # noise-immune knobs
        dp_precond_floor_factor: float = 1.0,
        dp_floor_mode: str = "geometry",
        precond_cond_max: Optional[float] = 1e4,
        precond_cond_strategy: str = "raise_small",
        precond_update_mode: str = "current",
        lift_gauge_fix: str = "none",
        moment_gauge_fix: str = "none",
        gauge_fix_eps: float = 1e-12,
        max_update_norm: float = 0.0,
        # LAMB-style trust ratio (intrinsic)
        use_trust_ratio: bool = False,
        trust_clip: Tuple[float, float] = (0.0, 10.0),
        trust_eps: float = 1e-12,
        # DP-aware Adam fix: debias v by subtracting expected DP noise covariance
        dp_debias_second_moment: bool = True,
        log_every: int = 1,
    ):
        if lr <= 0:
            raise ValueError("lr must be positive")
        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError("betas must be in [0,1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if trust_eps <= 0:
            raise ValueError("trust_eps must be positive")
        if trust_clip[1] < trust_clip[0]:
            raise ValueError("trust_clip max must be >= min")
        if trust_clip[0] < 0:
            raise ValueError("trust_clip min must be >= 0")
        valid_floor_modes = {"geometry", "scalar", "none"}
        if str(dp_floor_mode).lower() not in valid_floor_modes:
            raise ValueError(f"dp_floor_mode must be one of {sorted(valid_floor_modes)}, got {dp_floor_mode!r}")
        valid_precond_modes = {"current", "delayed"}
        if str(precond_update_mode).lower() not in valid_precond_modes:
            raise ValueError(f"precond_update_mode must be one of {sorted(valid_precond_modes)}, got {precond_update_mode!r}")
        valid_gauge_fix_modes = {"none", "pre_moment", "pre_retract", "both"}
        if str(lift_gauge_fix).lower() not in valid_gauge_fix_modes:
            raise ValueError(f"lift_gauge_fix must be one of {sorted(valid_gauge_fix_modes)}, got {lift_gauge_fix!r}")
        valid_moment_gauge_fix_modes = {"none", "hat", "state"}
        if str(moment_gauge_fix).lower() not in valid_moment_gauge_fix_modes:
            raise ValueError(f"moment_gauge_fix must be one of {sorted(valid_moment_gauge_fix_modes)}, got {moment_gauge_fix!r}")
        if gauge_fix_eps <= 0:
            raise ValueError("gauge_fix_eps must be positive")


        defaults = dict(lr=lr)
        super().__init__(params, defaults)

        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.rcond = float(rcond)
        self.lora_l_dim = int(lora_l_dim)
        self.lora_r_dim = int(lora_r_dim)
        self.use_adaptive = bool(use_adaptive)

        self.dp_precond_floor_factor = float(dp_precond_floor_factor)
        self.dp_floor_mode = str(dp_floor_mode).lower()
        self.precond_cond_max = float(precond_cond_max) if precond_cond_max is not None else None
        self.precond_cond_strategy = str(precond_cond_strategy)
        self.precond_update_mode = str(precond_update_mode).lower()
        self.lift_gauge_fix = str(lift_gauge_fix).lower()
        self.moment_gauge_fix = str(moment_gauge_fix).lower()
        self.gauge_fix_eps = float(gauge_fix_eps)
        self.max_update_norm = float(max_update_norm)
        self.log_every = int(log_every)

        self.use_trust_ratio = bool(use_trust_ratio)
        self.trust_clip = (float(trust_clip[0]), float(trust_clip[1]))
        self.trust_eps = float(trust_eps)
        self.dp_debias_second_moment = bool(dp_debias_second_moment)

        self._helper = _LoraShapeHelper()
        self._dp_state: Optional[_DPAccumState] = None
        self._state_initialized = False

        # user-readable per-step logs
        self.last_log: Dict[str, float] = {}

    # -------------------------
    # Parameter pairing (LoRA A/B are consecutive in the param list)
    # -------------------------

    def _pair_iter(self):
        for group in self.param_groups:
            params = group["params"]
            for p1, p2 in list(zip(params, params[1:]))[::2]:
                yield group, p1, p2

    def _get_factors(self, pA: torch.nn.Parameter, pB: torch.nn.Parameter) -> Tuple[Tensor, Tensor]:
        # pA is lora_A.weight (r,in) -> B (in,r)
        B, _ = self._helper.move_lora_dim_to_last(pA.data, self.lora_l_dim)
        # pB is lora_B.weight (out,r) -> A (out,r)
        A, _ = self._helper.move_lora_dim_to_last(pB.data, self.lora_r_dim)
        return A, B

    def _set_factors(self, pA: torch.nn.Parameter, pB: torch.nn.Parameter, A: Tensor, B: Tensor) -> None:
        pB.data.copy_(self._helper.restore_param_shape(A, pB.data, self.lora_r_dim))
        pA.data.copy_(self._helper.restore_param_shape(B, pA.data, self.lora_l_dim))

    def _ensure_state(self):
        if self._state_initialized:
            return
        for _, pA, pB in self._pair_iter():
            A, B = self._get_factors(pA, pB)
            r = A.shape[1]
            self.state[pA]["prism"] = types.SimpleNamespace(
                step=0,
                mA=torch.zeros_like(A, dtype=torch.float32),
                mB=torch.zeros_like(B, dtype=torch.float32),
                vA=torch.zeros((r, r), device=A.device, dtype=torch.float32),
                vB=torch.zeros((r, r), device=B.device, dtype=torch.float32),
                # DP accumulation
                dp_accum_A=None,
                dp_accum_B=None,
                dp_cache=None,
                # last spectrum stats
                last_spec_A={},
                last_spec_B={},
            )
        self._state_initialized = True

    # -------------------------
    # Tangent gradient (invariant) from factor grads
    # -------------------------

    def _compute_tangent_grad(
        self,
        A: Tensor,
        B: Tensor,
        gA: Tensor,
        gB: Tensor,
        M_pinv: Optional[Tensor] = None,
        N_pinv: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute (dA,dB) such that dZ = dA B^T + A dB^T = P(G)."""
        M = A.T @ A
        N = B.T @ B
        if M_pinv is None:
            M_pinv = _psd_pinv(M, rcond=self.rcond)
        if N_pinv is None:
            N_pinv = _psd_pinv(N, rcond=self.rcond)


        # Use QR-based orthogonal projectors for numerical stability.
        QA, _ = torch.linalg.qr(A.float(), mode="reduced")
        QB, _ = torch.linalg.qr(B.float(), mode="reduced")

        YA = gA @ N_pinv
        proj_YA = _apply_projector_Q(QA, YA)
        dA = YA - 0.5 * proj_YA

        YB = gB @ M_pinv
        proj_YB = _apply_projector_Q(QB, YB)
        dB = YB - 0.5 * proj_YB

        return dA, dB, M, N, M_pinv, N_pinv

    # -------------------------
    # Noise-immune adaptive direction
    # -------------------------

    def _adaptive_direction(
        self,
        st: types.SimpleNamespace,
        gradA: Tensor,  # (m,r) fp32
        gradB: Tensor,  # (n,r) fp32
        *,
        A_for_gauge: Optional[Tensor] = None,
        B_for_gauge: Optional[Tensor] = None,
        noise_floor: float = 0.0,
            noise_floor_A: Optional[float] = None,
            noise_floor_B: Optional[float] = None,
        debias_vA: Optional[Tensor] = None,
        debias_vB: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Return preconditioned directions uA,uB (same shapes), fp32."""
        if not self.use_adaptive:
            return gradA, gradB

        st.step += 1
        t = st.step
        b1, b2 = self.beta1, self.beta2

        # first moment
        st.mA.mul_(b1).add_(gradA, alpha=1.0 - b1)
        st.mB.mul_(b1).add_(gradB, alpha=1.0 - b1)

        # Optional horizontalization of the persistent first moment.
        # Even if every current DP update is represented by a horizontal lift,
        # the tangent space moves after retraction. Removing the vertical/gauge
        # component of the EMA relative to the current (A,B) is deterministic
        # post-processing and does not change the intrinsic tangent represented
        # by the moment.
        if self.moment_gauge_fix == "state" and A_for_gauge is not None and B_for_gauge is not None:
            st.mA, st.mB = _horizontalize_lift(
                st.mA, st.mB, A_for_gauge.float(), B_for_gauge.float(), eps=self.gauge_fix_eps
            )

        # second moment (rank-space covariance on the right).
        #
        # ``current`` is the Adam-style update used in the paper-code baseline:
        # the current DP-sanitized tangent update contributes to V_t before V_t^{-1/2}
        # is applied.
        #
        # ``delayed`` is a one-step stale preconditioner ablation: from step 2 onward
        # we apply V_{t-1}^{-1/2} to the current first moment and only then update
        # V_t.  This remains deterministic post-processing of DP-sanitized updates,
        # and reduces direct coupling between the current Gaussian draw and the
        # rank-space preconditioner.  Step 1 falls back to ``current`` to avoid a
        # floor-only preconditioner.
        m_dim = float(gradA.shape[0])
        n_dim = float(gradB.shape[0])
        vA_new = (gradA.T @ gradA) / max(1.0, m_dim)
        vB_new = (gradB.T @ gradB) / max(1.0, n_dim)

        # bias correction for first moment
        bc1 = 1.0 - (b1 ** t)
        mA_hat = st.mA / bc1
        mB_hat = st.mB / bc1

        # Less invasive variant: only horizontalize the bias-corrected moment
        # used in this step, without mutating the stored EMA.
        if self.moment_gauge_fix == "hat" and A_for_gauge is not None and B_for_gauge is not None:
            mA_hat, mB_hat = _horizontalize_lift(
                mA_hat, mB_hat, A_for_gauge.float(), B_for_gauge.float(), eps=self.gauge_fix_eps
            )

        use_delayed = (self.precond_update_mode == "delayed" and t > 1)
        if use_delayed:
            bc2_prev = 1.0 - (b2 ** (t - 1))
            vA_hat = st.vA / max(bc2_prev, 1e-12)
            vB_hat = st.vB / max(bc2_prev, 1e-12)
            # Update the stored second moment after constructing the stale
            # preconditioner for this step.
            st.vA.mul_(b2).add_(vA_new, alpha=1.0 - b2)
            st.vB.mul_(b2).add_(vB_new, alpha=1.0 - b2)
        else:
            st.vA.mul_(b2).add_(vA_new, alpha=1.0 - b2)
            st.vB.mul_(b2).add_(vB_new, alpha=1.0 - b2)
            bc2 = 1.0 - (b2 ** t)
            vA_hat = st.vA / bc2
            vB_hat = st.vB / bc2

        # Optional DP-aware debiasing: subtract expected DP noise covariance in rank space.
        # This reduces the tendency of Adam/RMSProp to "normalize DP noise to unit scale".
        if debias_vA is not None:
            vA_hat = _sym(vA_hat - debias_vA.to(device=vA_hat.device, dtype=vA_hat.dtype))
        if debias_vB is not None:
            vB_hat = _sym(vB_hat - debias_vB.to(device=vB_hat.device, dtype=vB_hat.dtype))

        # DP-aware floor: prevent invsqrt from exploding in low-signal regimes.
        nfA = float(noise_floor if noise_floor_A is None else noise_floor_A)
        nfB = float(noise_floor if noise_floor_B is None else noise_floor_B)

        PA, specA = _psd_invsqrt_clamped(vA_hat, eps=self.eps, floor=nfA, cond_max=self.precond_cond_max, cond_strategy=self.precond_cond_strategy)
        PB, specB = _psd_invsqrt_clamped(vB_hat, eps=self.eps, floor=nfB, cond_max=self.precond_cond_max, cond_strategy=self.precond_cond_strategy)
        st.last_spec_A = specA
        st.last_spec_B = specB

        uA = mA_hat @ PA
        uB = mB_hat @ PB
        return uA, uB

    # -------------------------
    # Optional post-preconditioner update clipping
    # -------------------------

    def _maybe_clip_update(self, dA: Tensor, dB: Tensor, A: Tensor, B: Tensor) -> Tuple[Tensor, Tensor, float]:
        cap = float(self.max_update_norm)
        if cap <= 0:
            return dA, dB, 1.0
        n2 = _tangent_fro_norm_sq(dA, dB, A, B)
        n = torch.sqrt(torch.clamp(n2, min=0.0) + 1e-12)
        coef = float(torch.clamp(torch.tensor(cap, device=n.device, dtype=n.dtype) / n, max=1.0).item())
        return dA * coef, dB * coef, coef

    def _compute_trust_ratio(self, A: Tensor, B: Tensor, dA: Tensor, dB: Tensor) -> float:
        """LAMB-style trust ratio in intrinsic coordinates.

        trust = ||Z||_F / (||dZ||_F + eps), clipped.
        This is a deterministic post-processing step (DP-safe) that often stabilizes
        training when gradients are dominated by DP noise.
        """
        if not self.use_trust_ratio:
            return 1.0

        z2 = _z_fro_norm_sq(A, B)
        u2 = _tangent_fro_norm_sq(dA, dB, A, B)
        z = torch.sqrt(torch.clamp(z2, min=0.0) + 1e-12)
        u = torch.sqrt(torch.clamp(u2, min=0.0) + 1e-12)

        if float(z.item()) <= float(self.trust_eps):
            return 1.0

        trust = z / (u + float(self.trust_eps))
        tmin, tmax = self.trust_clip
        trust = torch.clamp(trust, min=float(tmin), max=float(tmax))
        return float(trust.item())

    # -------------------------
    # Non-DP step
    # -------------------------

    @torch.no_grad()
    def step(self, closure=None):
        self._ensure_state()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        step_logs = {}

        for group, pA, pB in self._pair_iter():
            lr: float = float(group["lr"])
            st = self.state[pA]["prism"]

            A, B = self._get_factors(pA, pB)

            gB_raw = self._helper.get_grad_tensor(pA)
            gA_raw = self._helper.get_grad_tensor(pB)
            gB, _ = self._helper.move_lora_dim_to_last(gB_raw, self.lora_l_dim)
            gA, _ = self._helper.move_lora_dim_to_last(gA_raw, self.lora_r_dim)

            A32, B32 = A.float(), B.float()
            gA32, gB32 = gA.float(), gB.float()

            dA_grad, dB_grad, _, _, _, _ = self._compute_tangent_grad(A32, B32, gA32, gB32)

            # Non-DP path: honor the same lift_gauge_fix semantics as the DP path.
            # In EXP14 (lift_gauge_fix="both"), the DP path canonicalizes the
            # DP-sanitized tangent before updating rank-space moments and again
            # before retraction.  The previous non-DP step only did the second
            # canonicalization, so EXP14 was effectively "pre_retract" in non-DP.
            # This deterministic gauge-fix does not alter the represented
            # intrinsic tangent dZ; it only selects a horizontal/min-norm lift.
            if self.lift_gauge_fix in {"pre_moment", "both"}:
                dA_grad, dB_grad = _horizontalize_lift(
                    dA_grad, dB_grad, A32, B32, eps=self.gauge_fix_eps
                )

            uA, uB = self._adaptive_direction(st, dA_grad, dB_grad, A_for_gauge=A32, B_for_gauge=B32, noise_floor=0.0)

            if self.weight_decay > 0:
                uA = uA + self.weight_decay * A32
                uB = uB + self.weight_decay * B32

            dA_dir = -uA
            dB_dir = -uB

            if self.lift_gauge_fix in {"pre_retract", "both"}:
                dA_dir, dB_dir = _horizontalize_lift(dA_dir, dB_dir, A32, B32, eps=self.gauge_fix_eps)

            trust = self._compute_trust_ratio(A32, B32, dA_dir, dB_dir)
            dA_dir = dA_dir * trust
            dB_dir = dB_dir * trust

            dA_dir, dB_dir, upd_coef = self._maybe_clip_update(dA_dir, dB_dir, A32, B32)

            r = A32.shape[1]
            A_new, B_new = _retract_rank_r(A32, B32, dA_dir, dB_dir, eta=lr, r=r, align_to=(A32, B32))
            self._set_factors(pA, pB, A_new.to(A.dtype), B_new.to(B.dtype))

            step_logs.update({
                "nondp_update_clip_coef": float(upd_coef),
                "nondp_trust_ratio": float(trust),
                "nondp_precond_eigA_min": float(st.last_spec_A.get("eig_min_clamped", 0.0) or 0.0),
                "nondp_precond_eigA_max": float(st.last_spec_A.get("eig_max_clamped", 0.0) or 0.0),
            })

        self.last_log = step_logs
        return loss

    # -------------------------
    # DP pathway
    # -------------------------

    @torch.no_grad()
    def dp_begin(self, max_grad_norm: float) -> None:
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        self._ensure_state()
        self._dp_state = _DPAccumState(max_grad_norm=float(max_grad_norm), total_samples=0)

        for _, pA, pB in self._pair_iter():
            st = self.state[pA]["prism"]
            A, B = self._get_factors(pA, pB)
            A32 = A.float()
            B32 = B.float()

            # Orthonormal bases for stable projectors Π_A, Π_B
            QA, _ = torch.linalg.qr(A32, mode="reduced")
            QB, _ = torch.linalg.qr(B32, mode="reduced")

            M = (A32.T @ A32)
            N = (B32.T @ B32)
            M_pinv = _psd_pinv(M, rcond=self.rcond)
            N_pinv = _psd_pinv(N, rcond=self.rcond)

            # Invsqrt for noise injection (damped to avoid instability)
            M_invsqrt = _psd_invsqrt_damped(M, eps=self.eps)
            N_invsqrt = _psd_invsqrt_damped(N, eps=self.eps)

            st.dp_cache = {
                "A": A32,
                "B": B32,
                "QA": QA,
                "QB": QB,
                "M": M,
                "N": N,
                "M_pinv": M_pinv,
                "N_pinv": N_pinv,
                "M_invsqrt": M_invsqrt,
                "N_invsqrt": N_invsqrt,
            }
            st.dp_accum_A = torch.zeros_like(A32)
            st.dp_accum_B = torch.zeros_like(B32)

    @torch.no_grad()
    def dp_accumulate(self) -> int:
        """Accumulate clipped per-example tangent updates using global clipping.

        Global norm: ||dZ_i||^2 = sum_ell ||dZ_{i,ell}||_F^2
        Shared coef: alpha_i = min(1, C / ||dZ_i||)
        """
        if self._dp_state is None:
            raise RuntimeError("dp_begin() must be called before dp_accumulate().")

        max_norm = float(self._dp_state.max_grad_norm)

        total_in_micro: Optional[int] = None
        global_norm_sq: Optional[Tensor] = None

        # pass 1: compute per-sample global norm
        for _, pA, pB in self._pair_iter():
            st = self.state[pA]["prism"]
            cache = st.dp_cache
            if cache is None:
                raise RuntimeError("dp_cache missing; did you call dp_begin()?")

            gsA_raw = self._helper.get_grad_sample_tensor(pB)
            gsB_raw = self._helper.get_grad_sample_tensor(pA)
            if gsA_raw is None or gsB_raw is None:
                continue

            gA = self._helper.move_lora_dim_to_last_grad_sample(gsA_raw, self.lora_r_dim).float()  # (bs,m,r)
            gB = self._helper.move_lora_dim_to_last_grad_sample(gsB_raw, self.lora_l_dim).float()  # (bs,n,r)

            bs = int(gA.shape[0])
            total_in_micro = bs if total_in_micro is None else min(total_in_micro, bs)

            A = cache["A"]
            B = cache["B"]
            M_pinv = cache["M_pinv"]
            N_pinv = cache["N_pinv"]

            QA = cache["QA"]
            QB = cache["QB"]

            YA = torch.matmul(gA, N_pinv)
            projYA = _apply_projector_Q(QA, YA)
            dA = YA - 0.5 * projYA

            YB = torch.matmul(gB, M_pinv)
            projYB = _apply_projector_Q(QB, YB)
            dB = YB - 0.5 * projYB

            # per-sample ||dZ||^2
            n2 = _tangent_fro_norm_sq(dA, dB, A, B)
            if global_norm_sq is None:
                global_norm_sq = n2
            else:
                assert total_in_micro is not None
                global_norm_sq = global_norm_sq[:total_in_micro] + n2[:total_in_micro]

        if total_in_micro is None or global_norm_sq is None:
            return 0

        global_norm_sq = global_norm_sq[:total_in_micro]
        global_norm = torch.sqrt(torch.clamp(global_norm_sq, min=0.0) + 1e-12)
        coef = (max_norm / global_norm).clamp(max=1.0)

        # log microbatch clip stats
        clipped_frac = float((coef < 1.0).float().mean().item())
        coef_mean = float(coef.mean().item())
        coef_min = float(coef.min().item())

        self._dp_state.microbatches += 1
        self._dp_state.clipped_frac_sum += clipped_frac
        self._dp_state.coef_mean_sum += coef_mean
        self._dp_state.coef_min_sum += coef_min
        self._dp_state.last_micro_stats = {
            "micro_clipped_frac": clipped_frac,
            "micro_coef_mean": coef_mean,
            "micro_coef_min": coef_min,
            "micro_global_norm_mean": float(global_norm.mean().item()),
            "micro_global_norm_p95": float(torch.quantile(global_norm, 0.95).item()),
        }

        coef_view = coef.view(-1, 1, 1)

        # pass 2: apply shared coef and accumulate
        for _, pA, pB in self._pair_iter():
            st = self.state[pA]["prism"]
            cache = st.dp_cache
            if cache is None:
                raise RuntimeError("dp_cache missing; did you call dp_begin()?")

            gsA_raw = self._helper.get_grad_sample_tensor(pB)
            gsB_raw = self._helper.get_grad_sample_tensor(pA)
            if gsA_raw is None or gsB_raw is None:
                continue

            gA = self._helper.move_lora_dim_to_last_grad_sample(gsA_raw, self.lora_r_dim).float()
            gB = self._helper.move_lora_dim_to_last_grad_sample(gsB_raw, self.lora_l_dim).float()
            if int(gA.shape[0]) != total_in_micro:
                gA = gA[:total_in_micro]
            if int(gB.shape[0]) != total_in_micro:
                gB = gB[:total_in_micro]

            A = cache["A"]
            B = cache["B"]
            M_pinv = cache["M_pinv"]
            N_pinv = cache["N_pinv"]

            QA = cache["QA"]
            QB = cache["QB"]

            YA = torch.matmul(gA, N_pinv)
            projYA = _apply_projector_Q(QA, YA)
            dA = YA - 0.5 * projYA

            YB = torch.matmul(gB, M_pinv)
            projYB = _apply_projector_Q(QB, YB)
            dB = YB - 0.5 * projYB

            st.dp_accum_A.add_((dA * coef_view).sum(dim=0))
            st.dp_accum_B.add_((dB * coef_view).sum(dim=0))

            self._helper.clear_grad_sample(pA)
            self._helper.clear_grad_sample(pB)

        self._dp_state.total_samples += int(total_in_micro)
        return int(total_in_micro)

    @torch.no_grad()
    def dp_finalize(self, noise_multiplier: float) -> int:
        if self._dp_state is None:
            raise RuntimeError("dp_begin() must be called before dp_finalize().")

        total = int(self._dp_state.total_samples)
        if total <= 0:
            self._dp_state = None
            return 0

        C = float(self._dp_state.max_grad_norm)
        sigma = float(noise_multiplier)
        std = sigma * C / float(total)

        # aggregated logging across modules
        signal_norm_sq = 0.0
        noise_norm_sq = 0.0
        update_clip_coef_min = 1.0
        trust_ratio_min = float('inf')
        trust_ratio_max = 0.0
        precond_eigA_min = float('inf')
        precond_eigA_max = 0.0
        precond_eigB_min = float('inf')
        precond_eigB_max = 0.0
        floorA_min = float('inf')
        floorA_max = 0.0
        floorB_min = float('inf')
        floorB_max = 0.0

        for group, pA, pB in self._pair_iter():
            lr: float = float(group["lr"])
            st = self.state[pA]["prism"]
            cache = st.dp_cache
            if cache is None:
                continue

            A = cache["A"]
            B = cache["B"]
            M_pinv = cache["M_pinv"]
            N_pinv = cache["N_pinv"]
            M_invsqrt = cache["M_invsqrt"]
            N_invsqrt = cache["N_invsqrt"]
            QA = cache.get("QA")
            if QA is None:
                QA, _ = torch.linalg.qr(A, mode="reduced")

            gradA = st.dp_accum_A / float(total)
            gradB = st.dp_accum_B / float(total)

            # DP noise sampler implementing tangent isotropic Gaussian
            if std > 0:
                U = torch.randn_like(A)
                V = torch.randn_like(B)
                projU = _apply_projector_Q(QA, U)
                U_perp = U - projU
                noise_A = (U_perp @ N_invsqrt) * std
                noise_B = (V @ M_invsqrt) * std
            else:
                noise_A = torch.zeros_like(A)
                noise_B = torch.zeros_like(B)

            gradA_noisy = gradA + noise_A
            gradB_noisy = gradB + noise_B

            if self.lift_gauge_fix in {"pre_moment", "both"}:
                # Choose a canonical/minimum-norm lift of the same intrinsic
                # DP tangent before rank-space moments are updated.  This does
                # not alter dZ; it only removes lift-gauge degrees of freedom.
                gradA_noisy, gradB_noisy = _horizontalize_lift(
                    gradA_noisy, gradB_noisy, A, B, eps=self.gauge_fix_eps
                )

            # for logging: intrinsic norms of signal/noise (module-level)
            sig_n2 = _tangent_fro_norm_sq(gradA, gradB, A, B)
            noi_n2 = _tangent_fro_norm_sq(noise_A, noise_B, A, B)
            signal_norm_sq += float(sig_n2.item())
            noise_norm_sq += float(noi_n2.item())

            # DP-aware preconditioner floors.  ``geometry`` follows the rank-space
            # noise covariance suggested by the PRISM analysis.  ``scalar`` keeps the
            # same DP-noise-scale floor as the original paper-code implementation and
            # is useful when the geometry floor is too conservative on a particular run.
            base_floor = float(self.dp_precond_floor_factor) * float(std * std)
            if self.dp_floor_mode == "none":
                noise_floor_A = 0.0
                noise_floor_B = 0.0
            else:
                noise_floor_A = base_floor
                noise_floor_B = base_floor

            debias_vA = None
            debias_vB = None
            if std > 0:
                m_dim = float(A.shape[0])
                r_dim = float(A.shape[1])
                coefA = max(m_dim - r_dim, 0.0) / max(m_dim, 1.0)
                N_inv = _psd_inv_from_invsqrt(N_invsqrt)
                M_inv = _psd_inv_from_invsqrt(M_invsqrt)

                if self.dp_floor_mode == "geometry":
                    # scalar floors based on average eigenvalue of the expected
                    # tangent-noise covariance in rank space.
                    trN_inv = float(torch.trace(N_inv).item()) / max(r_dim, 1.0)
                    trM_inv = float(torch.trace(M_inv).item()) / max(r_dim, 1.0)
                    noise_floor_A = max(base_floor, float(self.dp_precond_floor_factor) * float(std * std) * float(coefA) * trN_inv)
                    noise_floor_B = max(base_floor, float(self.dp_precond_floor_factor) * float(std * std) * trM_inv)

                if self.dp_debias_second_moment:
                    debias_vA = (std * std) * float(coefA) * N_inv
                    debias_vB = (std * std) * M_inv

            floorA_min = min(floorA_min, float(noise_floor_A))
            floorA_max = max(floorA_max, float(noise_floor_A))
            floorB_min = min(floorB_min, float(noise_floor_B))
            floorB_max = max(floorB_max, float(noise_floor_B))

            # adaptive direction
            uA, uB = self._adaptive_direction(
                st,
                gradA_noisy,
                gradB_noisy,
                A_for_gauge=A,
                B_for_gauge=B,
                noise_floor=base_floor,
                noise_floor_A=noise_floor_A,
                noise_floor_B=noise_floor_B,
                debias_vA=debias_vA,
                debias_vB=debias_vB,
            )
            # aggregate preconditioner spectrum stats (diagnostics)
            precond_eigA_min = min(precond_eigA_min, float(st.last_spec_A.get("eig_min_clamped", 0.0) or 0.0))
            precond_eigA_max = max(precond_eigA_max, float(st.last_spec_A.get("eig_max_clamped", 0.0) or 0.0))
            precond_eigB_min = min(precond_eigB_min, float(st.last_spec_B.get("eig_min_clamped", 0.0) or 0.0))
            precond_eigB_max = max(precond_eigB_max, float(st.last_spec_B.get("eig_max_clamped", 0.0) or 0.0))

            if self.weight_decay > 0:
                uA = uA + self.weight_decay * A
                uB = uB + self.weight_decay * B

            dA_dir = -uA
            dB_dir = -uB

            if self.lift_gauge_fix in {"pre_retract", "both"}:
                dA_dir, dB_dir = _horizontalize_lift(dA_dir, dB_dir, A, B, eps=self.gauge_fix_eps)

            trust = self._compute_trust_ratio(A, B, dA_dir, dB_dir)
            dA_dir = dA_dir * trust
            dB_dir = dB_dir * trust
            trust_ratio_min = min(trust_ratio_min, float(trust))
            trust_ratio_max = max(trust_ratio_max, float(trust))

            dA_dir, dB_dir, upd_coef = self._maybe_clip_update(dA_dir, dB_dir, A, B)
            update_clip_coef_min = min(update_clip_coef_min, float(upd_coef))

            r = A.shape[1]
            A_new, B_new = _retract_rank_r(A, B, dA_dir, dB_dir, eta=lr, r=r, align_to=(A, B))
            self._set_factors(pA, pB, A_new.to(pB.data.dtype), B_new.to(pA.data.dtype))

            # clear dp buffers
            st.dp_cache = None
            st.dp_accum_A = None
            st.dp_accum_B = None

        # finalize log
        mb = max(1, int(self._dp_state.microbatches))
        clip_frac = self._dp_state.clipped_frac_sum / mb
        coef_mean = self._dp_state.coef_mean_sum / mb
        coef_min = self._dp_state.coef_min_sum / mb

        sig = math.sqrt(max(signal_norm_sq, 0.0))
        noi = math.sqrt(max(noise_norm_sq, 0.0))
        snr = float(sig / (noi + 1e-12))

        self.last_log = {
            "dp_total_samples": float(total),
            "dp_noise_multiplier": float(noise_multiplier),
            "dp_std_per_factor": float(std),
            "dp_floor_mode_id": float({"none": 0, "scalar": 1, "geometry": 2}.get(self.dp_floor_mode, -1)),
            "dp_precond_update_mode_id": float({"current": 0, "delayed": 1}.get(self.precond_update_mode, -1)),
            "dp_lift_gauge_fix_id": float({"none": 0, "pre_moment": 1, "pre_retract": 2, "both": 3}.get(self.lift_gauge_fix, -1)),
            "dp_moment_gauge_fix_id": float({"none": 0, "hat": 1, "state": 2}.get(self.moment_gauge_fix, -1)),
            "dp_floorA_min": float(0.0 if floorA_min == float("inf") else floorA_min),
            "dp_floorA_max": float(floorA_max),
            "dp_floorB_min": float(0.0 if floorB_min == float("inf") else floorB_min),
            "dp_floorB_max": float(floorB_max),
            "dp_clip_frac": float(clip_frac),
            "dp_coef_mean": float(coef_mean),
            "dp_coef_min": float(coef_min),
            "dp_signal_norm": float(sig),
            "dp_noise_norm": float(noi),
            "dp_snr": float(snr),
            "dp_update_clip_coef_min": float(update_clip_coef_min),
            "dp_trust_ratio_min": float(0.0 if trust_ratio_min == float("inf") else trust_ratio_min),
            "dp_trust_ratio_max": float(trust_ratio_max),
            "dp_precond_eigA_min": float(0.0 if precond_eigA_min == float("inf") else precond_eigA_min),
            "dp_precond_eigA_max": float(precond_eigA_max),
            "dp_precond_eigB_min": float(0.0 if precond_eigB_min == float("inf") else precond_eigB_min),
            "dp_precond_eigB_max": float(precond_eigB_max),
        }

        self._dp_state = None
        return total



# -----------------------------------------------------------------------------
# LoRA parameter pairing + no-Spectral full-rank initialization helpers
# -----------------------------------------------------------------------------


def get_paired_lora_parameters(
    model: torch.nn.Module,
    *,
    adapter_name: str = "default",
    require_grad_only: bool = True,
) -> List[torch.nn.Parameter]:
    """Return LoRA parameters in the exact order expected by PRISM.

    PRISM pairs parameters as (lora_A.weight, lora_B.weight).  Building the list
    from named_modules is safer than filtering model.named_parameters(), because
    it avoids accidental mis-pairing when PEFT/module ordering changes.
    """
    params: List[torch.nn.Parameter] = []
    for _, mod in model.named_modules():
        if not (hasattr(mod, "lora_A") and hasattr(mod, "lora_B")):
            continue
        if adapter_name not in getattr(mod, "lora_A") or adapter_name not in getattr(mod, "lora_B"):
            continue
        lora_A = mod.lora_A[adapter_name]
        lora_B = mod.lora_B[adapter_name]
        if not (hasattr(lora_A, "weight") and hasattr(lora_B, "weight")):
            continue
        pA = lora_A.weight
        pB = lora_B.weight
        if require_grad_only and (not pA.requires_grad or not pB.requires_grad):
            continue
        params.extend([pA, pB])
    if len(params) == 0:
        raise ValueError("No LoRA parameters found. Check adapter_name/target_modules.")
    if len(params) % 2 != 0:
        raise ValueError("Internal error: expected an even number of LoRA parameters.")
    return params


@torch.no_grad()
def balanced_full_rank_lora_init_peft_model(
    model: torch.nn.Module,
    *,
    adapter_name: str = "default",
    factor_scale: float = 2e-2,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Initialize both LoRA factors to tiny full-rank balanced factors.

    This is the recommended no-Spectral ablation for PRISM.  PEFT's default LoRA
    init usually sets lora_B=0, so A^T A is singular and the rank-r tangent
    formulas are not literally at a rank-r point.  Here both factors have rank r,
    while the initial intrinsic update remains tiny:

        singular values of (lora_B @ lora_A) are approximately factor_scale^2.

    No base-model weights are modified, unlike Spectral residual initialization.
    """
    if factor_scale <= 0:
        raise ValueError("factor_scale must be positive")

    gen = None
    if seed is not None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))

    touched = 0
    z_energy = 0.0

    for mod_name, mod in model.named_modules():
        if not (hasattr(mod, "lora_A") and hasattr(mod, "lora_B")):
            continue
        if adapter_name not in getattr(mod, "lora_A") or adapter_name not in getattr(mod, "lora_B"):
            continue

        lora_A = mod.lora_A[adapter_name]
        lora_B = mod.lora_B[adapter_name]
        if not (hasattr(lora_A, "weight") and hasattr(lora_B, "weight")):
            continue

        r = int(lora_A.weight.shape[0])
        in_features = int(lora_A.weight.shape[1])
        out_features = int(lora_B.weight.shape[0])
        if lora_B.weight.shape[1] != r:
            raise ValueError(f"LoRA shape mismatch in {mod_name}")

        # Draw on CPU for deterministic behavior across GPUs, then QR.
        G_in = torch.randn((in_features, r), generator=gen, dtype=torch.float32)
        G_out = torch.randn((out_features, r), generator=gen, dtype=torch.float32)
        Q_in, _ = torch.linalg.qr(G_in, mode="reduced")    # (in,r)
        Q_out, _ = torch.linalg.qr(G_out, mode="reduced")  # (out,r)

        # PEFT convention: lora_A.weight=(r,in), lora_B.weight=(out,r).
        B_T = (float(factor_scale) * Q_in.T).to(device=lora_A.weight.device, dtype=lora_A.weight.dtype)
        A = (float(factor_scale) * Q_out).to(device=lora_B.weight.device, dtype=lora_B.weight.dtype)

        lora_A.weight.data.copy_(B_T)
        lora_B.weight.data.copy_(A)

        # ||A B^T||_F^2 = r * factor_scale^4 for orthonormal Qs (before PEFT scaling).
        z_energy += float(r) * float(factor_scale) ** 4
        touched += 1

        if verbose and touched <= 3:
            print(f"[FullRankInit] {mod_name}: r={r} factor_scale={factor_scale:g}")

    stats = {
        "full_rank_init_modules": float(touched),
        "full_rank_init_factor_scale": float(factor_scale),
        "full_rank_init_z_energy_unscaled": float(z_energy),
    }
    if verbose:
        print(f"[FullRankInit] touched_modules={touched}, factor_scale={factor_scale:g}")
    return stats

# -----------------------------------------------------------------------------
# Spectral initialization helper
# -----------------------------------------------------------------------------


@torch.no_grad()
def randomized_svd(
    W: Tensor,
    rank: int,
    oversample: int = 8,
    n_iter: int = 2,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Randomized truncated SVD.

    Returns U (m,rank), S (rank,), Vh (rank,n) such that W ≈ U diag(S) Vh.

    This is substantially cheaper than full SVD for large matrices.
    """
    if W.ndim != 2:
        raise ValueError("W must be 2D")
    m, n = W.shape
    r = int(rank)
    l = min(n, r + int(oversample))
    Wf = W.float()

    Omega = torch.randn((n, l), device=W.device, dtype=Wf.dtype)
    Y = Wf @ Omega
    for _ in range(max(0, int(n_iter))):
        Y = Wf @ (Wf.T @ Y)

    Q, _ = torch.linalg.qr(Y, mode="reduced")
    B = Q.T @ Wf
    Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Ub

    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]
    return U_r.to(W.dtype), S_r.to(W.dtype), Vh_r.to(W.dtype)


@torch.no_grad()
def spectral_init_peft_model(
    model: torch.nn.Module,
    *,
    adapter_name: str = "default",
    svd_rank: Optional[int] = None,
    svd_oversample: int = 8,
    svd_n_iter: int = 2,
    svd_device: str = "cpu",
    verbose: bool = True,
    store_cache: bool = True,
    cache_dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    """Apply Spectral init to PEFT LoRA modules in-place.

    For each LoRA-augmented Linear weight W (out,in):
      - compute top-r SVD: W_r = U diag(S) V^T
      - set base weight <- W - W_r  (residual is frozen)
      - set LoRA weights so that (scaling * (B@A)) == W_r

    This keeps the *effective* initial weight unchanged.

    Returns summary stats.
    """
    cache: Dict[str, Dict[str, Tensor]] = {}
    touched = 0
    total_energy = 0.0
    kept_energy = 0.0

    for mod_name, mod in model.named_modules():
        if not (hasattr(mod, "lora_A") and hasattr(mod, "lora_B") and hasattr(mod, "weight")):
            continue
        if adapter_name not in getattr(mod, "lora_A"):
            continue

        lora_A = mod.lora_A[adapter_name]
        lora_B = mod.lora_B[adapter_name]
        if not (hasattr(lora_A, "weight") and hasattr(lora_B, "weight")):
            continue

        # LoRA rank
        r = int(lora_A.weight.shape[0])
        if svd_rank is not None:
            r = min(r, int(svd_rank))

        # scaling (PEFT stores a dict in recent versions)
        scaling = getattr(mod, "scaling", None)
        if isinstance(scaling, dict):
            scale = float(scaling.get(adapter_name, 1.0))
        else:
            scale = float(scaling) if scaling is not None else 1.0
        if scale <= 0:
            scale = 1.0

        W = mod.weight.data

        # move to SVD device (usually CPU) for stability/memory
        if svd_device == "cpu":
            W_svd = W.detach().to("cpu", dtype=torch.float32)
            U, S, Vh = randomized_svd(W_svd, rank=r, oversample=svd_oversample, n_iter=svd_n_iter)
            U = U.to(W.device, dtype=torch.float32)
            S = S.to(W.device, dtype=torch.float32)
            Vh = Vh.to(W.device, dtype=torch.float32)
        else:
            W_svd = W.detach().to(dtype=torch.float32)
            U, S, Vh = randomized_svd(W_svd, rank=r, oversample=svd_oversample, n_iter=svd_n_iter)
            U = U.to(dtype=torch.float32)
            S = S.to(dtype=torch.float32)
            Vh = Vh.to(dtype=torch.float32)

        # low-rank reconstruction
        Wr = (U * S.unsqueeze(0)) @ Vh

        # energy stats
        total_energy += float((W.float() ** 2).sum().item())
        kept_energy += float((Wr ** 2).sum().item())

        # keep effective initial weights unchanged: W <- W - Wr
        mod.weight.data = (W.float() - Wr).to(W.dtype)

        # set LoRA weights such that scale * (lora_B @ lora_A) == Wr
        # i.e., (lora_B @ lora_A) == Wr / scale
        S_scaled = S / float(scale)
        s_sqrt = torch.sqrt(torch.clamp(S_scaled, min=0.0))  # (r,)

        A_init = (U * s_sqrt.unsqueeze(0)).to(lora_B.weight.dtype)          # (out,r)
        B_init_T = (Vh * s_sqrt.unsqueeze(1)).to(lora_A.weight.dtype)       # (r,in)

        # Assign
        if lora_B.weight.data.shape != A_init.shape:
            raise ValueError(f"Spectral shape mismatch for {mod_name}: lora_B {tuple(lora_B.weight.shape)} vs A_init {tuple(A_init.shape)}")
        if lora_A.weight.data.shape != B_init_T.shape:
            raise ValueError(f"Spectral shape mismatch for {mod_name}: lora_A {tuple(lora_A.weight.shape)} vs B_init_T {tuple(B_init_T.shape)}")

        lora_B.weight.data.copy_(A_init)
        lora_A.weight.data.copy_(B_init_T)

        if store_cache:
            # Store the Spectral low-rank piece in factor form so we can later rebase
            # the adapter for saving/loading without requiring the modified base weights.
            cache[mod_name] = {
                "lora_B_init": A_init.detach().to("cpu", dtype=cache_dtype).contiguous(),
                "lora_A_init": B_init_T.detach().to("cpu", dtype=cache_dtype).contiguous(),
                "scale": float(scale),
            }

        touched += 1
        if verbose and touched <= 3:
            print(f"[Spectral] init {mod_name}: r={r} scale={scale:.6g} ||Wr||F={Wr.norm().item():.4g}")

    stats = {
        "spectral_modules": float(touched),
        "spectral_energy_total": float(total_energy),
        "spectral_energy_kept": float(kept_energy),
        "spectral_energy_ratio": float(kept_energy / (total_energy + 1e-12)),
    }
    if verbose:
        print(f"[Spectral] touched_modules={touched} kept_energy_ratio={stats['spectral_energy_ratio']:.4f}")
    if store_cache:
        # attach to the model so training/eval code can rebase adapters before saving
        setattr(model, "_spectral_cache", cache)
        setattr(model, "_spectral_adapter_name", adapter_name)

    return stats


@torch.no_grad()
def spectral_rebase_adapter_inplace(
    model: torch.nn.Module,
    *,
    adapter_name: str = "default",
    rank_multiplier: int = 2,
    verbose: bool = True,
) -> Dict[str, float]:
    """Rebase a Spectral-residual-trained adapter so it can be loaded on the *original* base model.

    Problem
    -------
    Spectral residual init modifies each frozen base weight W -> W - W_r and initializes
    LoRA so that scaling*(B@A) == W_r, keeping the effective weight unchanged.

    If you later save only the adapter and reload it on the original base model W,
    you would get W + Z instead of (W - W_r) + Z, which is wrong and can yield
    near-zero accuracy.

    Fix
    ---
    We convert the saved LoRA update to be relative to the *original* base weights by
    folding the fixed Spectral low-rank piece into the adapter itself:

        effective_train = (W - W_r) + Z
        effective_eval  = W + (Z - W_r)

    The matrix (Z - W_r) can have rank up to 2r, so by default we set the adapter
    rank to (rank_multiplier * r) and represent the sum exactly via concatenation
    of factors. This is a deterministic post-processing step (DP-safe).

    Returns summary stats.
    """

    cache = getattr(model, "_spectral_cache", None)
    if not isinstance(cache, dict) or len(cache) == 0:
        raise ValueError("No Spectral cache found on model. Run spectral_init_peft_model(..., store_cache=True) first.")

    if rank_multiplier < 1:
        raise ValueError("rank_multiplier must be >= 1")
    if rank_multiplier == 1:
        # nothing to do
        return {"spectral_rebase": 0.0, "spectral_rebase_modules": 0.0}

    touched = 0
    new_r_val: Optional[int] = None
    scale_val: Optional[float] = None

    for mod_name, mod in model.named_modules():
        if mod_name not in cache:
            continue
        if not (hasattr(mod, "lora_A") and hasattr(mod, "lora_B")):
            continue
        if adapter_name not in getattr(mod, "lora_A"):
            continue

        lora_A = mod.lora_A[adapter_name]
        lora_B = mod.lora_B[adapter_name]
        if not (hasattr(lora_A, "weight") and hasattr(lora_B, "weight")):
            continue

        A0_cpu = cache[mod_name]["lora_B_init"]  # (out,r)
        B0T_cpu = cache[mod_name]["lora_A_init"]  # (r,in)
        scale = float(cache[mod_name].get("scale", 1.0))

        # current
        B_curr_T = lora_A.weight.data
        A_curr = lora_B.weight.data

        r = int(B_curr_T.shape[0])
        if A_curr.shape[1] != r:
            raise ValueError(f"Unexpected LoRA shapes for {mod_name}: lora_A {tuple(B_curr_T.shape)}, lora_B {tuple(A_curr.shape)}")

        # move init factors to device/dtype
        A0 = A0_cpu.to(device=A_curr.device, dtype=A_curr.dtype)
        B0T = B0T_cpu.to(device=B_curr_T.device, dtype=B_curr_T.dtype)

        if A0.shape != A_curr.shape or B0T.shape != B_curr_T.shape:
            raise ValueError(
                f"Spectral cache shape mismatch for {mod_name}: "
                f"A0 {tuple(A0.shape)} vs A {tuple(A_curr.shape)}, "
                f"B0T {tuple(B0T.shape)} vs B {tuple(B_curr_T.shape)}"
            )

        new_r = int(r * rank_multiplier)
        if new_r_val is None:
            new_r_val = new_r
            scale_val = scale
        else:
            if new_r != new_r_val:
                raise ValueError("Rank mismatch across modules during Spectral rebase. This helper expects a uniform LoRA rank.")

        if rank_multiplier != 2:
            raise NotImplementedError("Only rank_multiplier=2 is currently supported.")

        # Construct exact factors for (Z - W_r):
        #   lora_B_new @ lora_A_new = (A_curr @ B_curr_T) + (-A0 @ B0T)
        B_new_T = torch.cat([B_curr_T, B0T], dim=0)          # (2r, in)
        A_new = torch.cat([A_curr, -A0], dim=1)              # (out, 2r)

        # Replace LoRA modules with the enlarged-rank versions
        in_features = int(getattr(lora_A, "in_features"))
        out_features = int(getattr(lora_B, "out_features"))

        new_lora_A = torch.nn.Linear(in_features, new_r, bias=False).to(device=B_curr_T.device, dtype=B_curr_T.dtype)
        new_lora_B = torch.nn.Linear(new_r, out_features, bias=False).to(device=A_curr.device, dtype=A_curr.dtype)

        new_lora_A.weight.data.copy_(B_new_T)
        new_lora_B.weight.data.copy_(A_new)

        mod.lora_A[adapter_name] = new_lora_A
        mod.lora_B[adapter_name] = new_lora_B

        # Update rank/alpha/scaling bookkeeping on the layer if present
        if hasattr(mod, "r"):
            if isinstance(mod.r, dict):
                mod.r[adapter_name] = new_r
            else:
                mod.r = new_r
        if hasattr(mod, "lora_alpha"):
            alpha_new = float(scale) * float(new_r)
            # Prefer integer alpha when it is effectively integral (for PEFT config compatibility)
            if abs(alpha_new - round(alpha_new)) < 1e-6:
                alpha_new = int(round(alpha_new))
            if isinstance(mod.lora_alpha, dict):
                mod.lora_alpha[adapter_name] = alpha_new
            else:
                mod.lora_alpha = alpha_new
        if hasattr(mod, "scaling"):
            if isinstance(mod.scaling, dict):
                mod.scaling[adapter_name] = float(scale)
            else:
                mod.scaling = float(scale)

        touched += 1

    # Update PEFT config (so loading constructs the enlarged ranks correctly)
    cfg = getattr(model, "peft_config", None)
    if isinstance(cfg, dict) and adapter_name in cfg:
        cfg_obj = cfg[adapter_name]
        # Some versions use rank_pattern/alpha_pattern; we clear them for simplicity
        if hasattr(cfg_obj, "rank_pattern"):
            cfg_obj.rank_pattern = {}
        if hasattr(cfg_obj, "alpha_pattern"):
            cfg_obj.alpha_pattern = {}
        if hasattr(cfg_obj, "r") and new_r_val is not None:
            cfg_obj.r = int(new_r_val)
        if hasattr(cfg_obj, "lora_alpha") and new_r_val is not None and scale_val is not None:
            # keep the same scaling: scaling = lora_alpha / r
            alpha_new = float(scale_val) * float(new_r_val)
            if abs(alpha_new - round(alpha_new)) < 1e-6:
                alpha_new = int(round(alpha_new))
            cfg_obj.lora_alpha = alpha_new

    if verbose:
        print(f"[Spectral] rebased adapter for saving/loading: touched_modules={touched}, new_rank={new_r_val}, scale={scale_val}")

    return {
        "spectral_rebase_modules": float(touched),
        "spectral_rebase_new_rank": float(0.0 if new_r_val is None else new_r_val),
        "spectral_rebase_scale": float(1.0 if scale_val is None else scale_val),
    }
