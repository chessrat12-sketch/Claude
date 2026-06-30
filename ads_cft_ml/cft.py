"""
CFT₂ (2차원 등각장론) 관련 계산

핵심 대상:
  - 1차 연산자(primary operator): 홀로그래픽 차원 Δ ↔ 벌크 질량 m
  - 응력 에너지 텐서 T(z): 중심 전하 c 로 특성화
  - OPE (Operator Product Expansion)
  - Virasoro 대수
  - 분배함수 Z(τ) - 모듈러 불변성
"""
import numpy as np
from typing import Tuple, Optional


# ── 기본 CFT 관계 ─────────────────────────────────────────────

def operator_dimension_from_mass(m: float, L: float = 1.0, d: int = 1) -> Tuple[float, float]:
    """
    AdS/CFT 대응: 벌크 장 질량 m → 경계 연산자 차원 Δ.

    m²L² = Δ(Δ - d)  →  Δ = d/2 ± √((d/2)² + m²L²)
    d = CFT 차원 (AdS_{d+1}에서)

    BF bound: m² ≥ -(d/2)²/L² (tachyons 허용)
    반환: (Δ₊, Δ₋) — 비정규화/정규화 모드
    """
    discriminant = (d / 2.0) ** 2 + m ** 2 * L ** 2
    if discriminant < 0:
        raise ValueError(f"BF bound 위반: m²L² = {m**2 * L**2:.3f} < {-(d/2)**2:.3f}")
    sqrt_disc = np.sqrt(discriminant)
    delta_plus = d / 2.0 + sqrt_disc
    delta_minus = d / 2.0 - sqrt_disc
    return delta_plus, delta_minus


def conformal_weight_from_dimension(delta: float, spin: int = 0) -> Tuple[float, float]:
    """
    등각 차원 Δ와 스핀 s → 홀로모픽/반홀로모픽 가중치 (h, h̄).
    h = (Δ + s) / 2,  h̄ = (Δ - s) / 2
    """
    h = (delta + spin) / 2.0
    h_bar = (delta - spin) / 2.0
    return h, h_bar


# ── 상관 함수 ─────────────────────────────────────────────────

def two_point_function(z1: complex, z2: complex, delta: float) -> float:
    """
    진공 2점 함수: <O(z₁) O(z₂)> = C / |z₁ - z₂|^{2Δ}
    Δ = 연산자의 (홀로모픽 + 반홀로모픽) 차원
    """
    dz = abs(z1 - z2)
    if dz < 1e-15:
        return np.inf
    return dz ** (-2 * delta)


def three_point_function(
    z1: complex, z2: complex, z3: complex,
    delta1: float, delta2: float, delta3: float,
    C123: float = 1.0,
) -> float:
    """
    3점 함수 (등각 불변성에 의해 결정):
    <O₁(z₁) O₂(z₂) O₃(z₃)> = C₁₂₃ / |z₁₂|^{a} |z₁₃|^{b} |z₂₃|^{c}
    a = Δ₁+Δ₂-Δ₃, b = Δ₁+Δ₃-Δ₂, c = Δ₂+Δ₃-Δ₁
    """
    z12 = abs(z1 - z2)
    z13 = abs(z1 - z3)
    z23 = abs(z2 - z3)
    a = delta1 + delta2 - delta3
    b = delta1 + delta3 - delta2
    c = delta2 + delta3 - delta1
    denom = (z12 ** a) * (z13 ** b) * (z23 ** c)
    return C123 / (denom + 1e-15)


def stress_tensor_ope(c: float) -> dict:
    """
    응력 텐서 OPE T(z)T(w) 계수.
    T(z)T(w) ~ c/2/(z-w)⁴ + 2T(w)/(z-w)² + ∂T(w)/(z-w) + ...
    """
    return {"c_over_2": c / 2.0, "dimension_T": 2, "spin_T": 2}


# ── Virasoro 대수 ──────────────────────────────────────────────

def virasoro_character(q: complex, h: float, c: float) -> complex:
    """
    Virasoro 표현의 지표 (character) — Verma module 경우.
    χ_h(q) = q^{h - c/24} / η(q)
    η(q) = q^{1/24} ∏_{n=1}^∞ (1-qⁿ)  [Dedekind η 함수 근사]
    """
    q = complex(q)
    eta = np.prod([1 - q ** n for n in range(1, 50)])
    prefactor = q ** (h - c / 24.0)
    return prefactor / eta


def modular_invariant_partition_function(
    tau: complex, c: float, spectrum: list
) -> complex:
    """
    모듈러 불변 분배함수 Z(τ) = Σ_{h,h̄} N_{h,h̄} χ_h(q) χ̄_h̄(q̄)
    q = e^{2πiτ}

    spectrum: list of (h, h_bar, multiplicity) 튜플
    """
    q = np.exp(2j * np.pi * tau)
    q_bar = np.conj(q)
    Z = 0.0 + 0.0j
    for h, h_bar, N in spectrum:
        chi_h = virasoro_character(q, h, c)
        chi_h_bar = virasoro_character(q_bar, h_bar, c)
        Z += N * chi_h * np.conj(chi_h_bar)
    return Z


# ── c-정리와 RG 흐름 ──────────────────────────────────────────

def zamolodchikov_c_function(g: np.ndarray, beta_fn_vals: np.ndarray) -> float:
    """
    Zamolodchikov c-함수 (2D). RG 흐름 따라 단조 감소.

    c(μ) = c_UV - 12π ∫_{μ}^{Λ} β_i(g) G^{ij} β_j(g) d(log μ')
    여기서 G^{ij} = Zamolodchikov 계량.

    단순화: 결합 상수 g 공간에서 궤적의 거리로 근사.
    """
    integrand = np.sum(beta_fn_vals ** 2, axis=-1)
    if len(integrand) < 2:
        return 0.0
    return float(np.trapz(integrand))


def central_charge_rg(c_uv: float, g_trajectory: np.ndarray) -> np.ndarray:
    """
    UV에서 IR로 RG 흐름 따라 유효 c-함수.
    Zamlodchikov의 c-정리: c는 RG 흐름 따라 단조 감소.
    """
    # 단순 모형: c(t) = c_UV * exp(-||Δg||²)
    delta_g = g_trajectory - g_trajectory[0]
    norm_sq = np.sum(delta_g ** 2, axis=-1)
    return c_uv * np.exp(-norm_sq)


# ── 홀로그래픽 사전(dictionary) ────────────────────────────────

HOLOGRAPHIC_DICTIONARY = {
    "boundary_source": "벌크 비정규화 모드 φ⁽⁰⁾(x) = lim_{z→0} z^{-Δ₊} φ(z,x)",
    "vev": "벌크 정규화 모드 φ⁽¹⁾(x) = lim_{z→0} z^{Δ₊} φ(z,x)",
    "partition_function": "Z_CFT[J] = exp(-S_AdS[φ_cl])",
    "stress_tensor": "경계 에너지-운동량 텐서 ↔ 벌크 계량 변동 δg_μν",
    "entanglement_entropy": "S = Area(γ_A)/(4G_N)  [Ryu-Takayanagi]",
    "thermal_state": "열 CFT 밀도 행렬 ↔ BTZ 블랙홀",
    "rg_flow": "CFT RG 흐름 ↔ 홀로그래픽 radial 방향 z",
}
