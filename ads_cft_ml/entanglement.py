"""
얽힘 엔트로피: 홀로그래픽(RT) vs CFT 해석적 결과 비교

Calabrese-Cardy 공식 (2004):
  진공: S = (c/3) log(l/ε)
  유한 온도: S = (c/3) log[(β/πε) sinh(πl/β)]
  유한 크기: S = (c/3) log[(L/πε) sin(πl/L)]

Ryu-Takayanagi (2006):
  S = Length(γ_A) / (4G_N)
  AdS₃에서 γ_A = 측지선 (반원) → 위 공식과 정확히 일치
"""
import numpy as np
from typing import List, Tuple, Optional
from .geometry import AdS3


# ── CFT 해석적 공식 ────────────────────────────────────────────

def ee_vacuum(l: float, c: float, epsilon: float = 1e-3) -> float:
    """진공 CFT₂ 얽힘 엔트로피. Calabrese-Cardy."""
    return (c / 3.0) * np.log(l / epsilon)


def ee_finite_temperature(l: float, c: float, beta: float, epsilon: float = 1e-3) -> float:
    """
    유한 온도 T = 1/β 에서의 얽힘 엔트로피.
    홀로그래픽 쌍대: BTZ 블랙홀에서의 RT 측지선.
    """
    return (c / 3.0) * np.log(beta / (np.pi * epsilon) * np.sinh(np.pi * l / beta))


def ee_finite_size(l: float, c: float, L_total: float, epsilon: float = 1e-3) -> float:
    """
    원 위(length L_total)의 CFT, 구간 l 의 얽힘 엔트로피.
    홀로그래픽 쌍대: global AdS₃ (열대적 방법).
    """
    return (c / 3.0) * np.log(L_total / (np.pi * epsilon) * np.sin(np.pi * l / L_total))


# ── 홀로그래픽 공식 (RT) ──────────────────────────────────────

def holographic_ee(adspace: AdS3, l: float, epsilon: float = 1e-3) -> float:
    """RT 공식으로 구한 홀로그래픽 얽힘 엔트로피."""
    return adspace.ryu_takayanagi_entropy(-l / 2.0, l / 2.0, epsilon)


def compare_rt_cft(adspace: AdS3, l_values: np.ndarray, epsilon: float = 1e-3) -> dict:
    """
    RT와 CFT 결과를 비교.
    두 결과는 c = 3L/(2G_N) 일 때 정확히 일치.
    """
    c = adspace.central_charge
    S_cft = ee_vacuum(l_values, c, epsilon)
    S_rt = np.array([holographic_ee(adspace, l, epsilon) for l in l_values])
    return {
        "l": l_values,
        "S_CFT": S_cft,
        "S_RT": S_rt,
        "relative_error": np.abs(S_cft - S_rt) / np.abs(S_cft + 1e-15),
        "c": c,
    }


# ── 상호 정보량과 홀로그래픽 위상 전이 ────────────────────────

def mutual_information_holographic(
    adspace: AdS3,
    A: Tuple[float, float],
    B: Tuple[float, float],
    epsilon: float = 1e-3,
) -> Tuple[float, str]:
    """
    두 구간 A=[a₁,a₂], B=[b₁,b₂] 사이의 홀로그래픽 상호 정보량.
    I(A:B) = S(A) + S(B) - S(A∪B)

    홀로그래픽 위상 전이:
    - 연결(connected) RT 면: γ_{A∪B} = γ_{a₁b₂} ∪ γ_{a₂b₁}
    - 비연결(disconnected) RT 면: γ_{A∪B} = γ_A ∪ γ_B
    두 면 중 길이가 짧은 것이 우세.
    """
    a1, a2 = A
    b1, b2 = B
    assert a2 < b1, "구간 A, B는 겹치지 않아야 함"

    S_A = adspace.ryu_takayanagi_entropy(a1, a2, epsilon)
    S_B = adspace.ryu_takayanagi_entropy(b1, b2, epsilon)

    # S(A∪B): 두 후보 RT 면
    S_conn = (
        adspace.ryu_takayanagi_entropy(a1, b2, epsilon)
        + adspace.ryu_takayanagi_entropy(a2, b1, epsilon)
    )
    S_disconn = S_A + S_B

    if S_conn < S_disconn:
        S_AuB = S_conn
        phase = "connected"
    else:
        S_AuB = S_disconn
        phase = "disconnected"

    I = S_A + S_B - S_AuB
    return max(I, 0.0), phase


def phase_transition_parameter(A_len: float, B_len: float, gap: float) -> float:
    """
    홀로그래픽 위상 전이 조건 (대칭인 경우 |A|=|B|=l):
    연결 ↔ 비연결 전이는 gap/l = log(√5 - 2) ≈ 0.48... 에서 발생.
    반환: x_ratio = gap / (A_len + B_len + gap)  (0~1)
    """
    total = A_len + B_len + gap
    return gap / total


def entanglement_spectrum(S: float, n_levels: int = 10) -> np.ndarray:
    """
    주어진 얽힘 엔트로피 S로부터 단순화된 얽힘 스펙트럼 추정.
    (등간격 분포 가정: λ_i ∝ e^{-αi})
    """
    alpha = 1.0 / n_levels  # 단순화
    lambdas = np.exp(-alpha * np.arange(n_levels))
    lambdas /= lambdas.sum()
    # 엔트로피 일치하도록 스케일 조정
    S_est = -np.sum(lambdas * np.log(lambdas + 1e-15))
    scale = S / S_est if S_est > 0 else 1.0
    # 재정규화
    lambdas = lambdas ** (1.0 / scale) if scale != 1.0 else lambdas
    lambdas /= lambdas.sum()
    return lambdas
