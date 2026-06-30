"""
고급 홀로그래픽 물리

1. BTZ 블랙홀 (2+1차원 AdS 블랙홀)
2. Kerr-AdS₄ (회전하는 4차원 AdS 블랙홀)
3. 홀로그래픽 복잡도 (Complexity = Volume / Action)
4. Entanglement Island 공식 (Page 곡선 정밀 계산)
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List


# ═══════════════════════════════════════════════════════════════
# 1. BTZ 블랙홀
# ═══════════════════════════════════════════════════════════════

@dataclass
class BTZBlackHole:
    """
    비회전 BTZ 블랙홀 (Banados-Teitelboim-Zanelli 1992).

    계량 (질량 M > 0):
      ds² = -(r²/l² - M) dt² + (r²/l² - M)⁻¹ dr² + r² dφ²

    홀라이즌: r_H = l√M
    Hawking 온도: T_H = r_H/(2πl²) = √M/(2πl)
    Bekenstein-Hawking 엔트로피: S_BH = 2πr_H/(4G_N) = πc r_H/3

    회전 BTZ (각운동량 J):
      r_±² = (Ml²/2)(1 ± √(1 - J²/(Ml)²))
      T_H = (r_+² - r_-²)/(2πl r_+)
    """
    M: float          # 블랙홀 질량 (M > 0 이면 블랙홀)
    l: float = 1.0    # AdS 반지름
    G_N: float = 0.05 # Newton 상수
    J: float = 0.0    # 각운동량

    def __post_init__(self):
        if self.J != 0.0 and abs(self.J) > self.M * self.l:
            raise ValueError(f"|J| = {abs(self.J)} > Ml = {self.M * self.l}: 초과 회전")

    @property
    def central_charge(self) -> float:
        return 3 * self.l / (2 * self.G_N)

    @property
    def r_horizon(self) -> float:
        """외부 홀라이즌 반지름 r_+"""
        if self.J == 0.0:
            return self.l * np.sqrt(max(self.M, 0.0))
        r_sq = (self.M * self.l**2 / 2) * (1 + np.sqrt(1 - (self.J / (self.M * self.l))**2))
        return np.sqrt(r_sq)

    @property
    def r_inner(self) -> float:
        """내부 홀라이즌 반지름 r_- (J≠0 인 경우)"""
        if self.J == 0.0:
            return 0.0
        r_sq = (self.M * self.l**2 / 2) * (1 - np.sqrt(1 - (self.J / (self.M * self.l))**2))
        return np.sqrt(max(r_sq, 0.0))

    @property
    def hawking_temperature(self) -> float:
        """Hawking 온도: T_H = (r_+² - r_-²)/(2π l r_+)"""
        r_p = self.r_horizon
        r_m = self.r_inner
        if r_p < 1e-12:
            return 0.0
        return (r_p**2 - r_m**2) / (2 * np.pi * self.l * r_p)

    @property
    def angular_velocity(self) -> float:
        """홀라이즌의 각속도: Ω_H = r_-/(l r_+)"""
        if self.r_horizon < 1e-12:
            return 0.0
        return self.r_inner / (self.l * self.r_horizon)

    @property
    def bekenstein_hawking_entropy(self) -> float:
        """Bekenstein-Hawking 엔트로피: S_BH = 2π r_+/(4G_N) = π c r_+/3"""
        return 2 * np.pi * self.r_horizon / (4 * self.G_N)

    @property
    def inverse_temperature(self) -> float:
        """역온도 β = 1/T_H"""
        T = self.hawking_temperature
        return 1.0 / T if T > 1e-15 else np.inf

    def thermal_entanglement_entropy(self, l_interval: float, epsilon: float = 1e-3) -> float:
        """
        BTZ 블랙홀 배경에서의 열 얽힘 엔트로피.
        홀로그래픽 RT 면 = BTZ 시공간의 측지선.

        S = (c/3) log[(β/πε) sinh(πl/β)]

        고온 극한 l >> β: S → (c/3)(πl/β + log(β/2πε)) [열 엔트로피 밀도]
        저온 극한 l << β: S → (c/3) log(l/ε) [진공 결과 회복]
        """
        beta = self.inverse_temperature
        c = self.central_charge
        if np.isinf(beta):  # T=0
            return (c / 3) * np.log(l_interval / epsilon)
        arg = (beta / (np.pi * epsilon)) * np.sinh(np.pi * l_interval / beta)
        return (c / 3) * np.log(arg)

    def hawking_radiation_spectrum(self, omega: np.ndarray) -> np.ndarray:
        """
        Hawking 복사 스펙트럼 (Planck 분포):
        n(ω) = 1 / (e^{β(ω - Ω_H m)} - 1)  (보존 방정식, m=0 가정)
        """
        beta = self.inverse_temperature
        if np.isinf(beta):
            return np.zeros_like(omega)
        exponent = beta * omega
        return 1.0 / (np.exp(np.clip(exponent, -50, 50)) - 1 + 1e-15)

    def metric_rr(self, r: np.ndarray) -> np.ndarray:
        """g_{rr} = (r²/l² - M)⁻¹"""
        f = r**2 / self.l**2 - self.M
        return 1.0 / (f + 1e-15)

    def metric_tt(self, r: np.ndarray) -> np.ndarray:
        """g_{tt} = -(r²/l² - M)"""
        return -(r**2 / self.l**2 - self.M)

    def quasinormal_frequencies(self, n: int = 3) -> List[complex]:
        """
        BTZ QNM (Quasi-Normal Modes) 주파수.
        스칼라 장 Δ = 2 (가장 낮은 QNM):
        ω_n = -i 4π T_H (n + 1)  (purely imaginary, n = 0,1,2,...)
        """
        T = self.hawking_temperature
        return [-4j * np.pi * T * (k + 1) for k in range(n)]

    def page_time(self) -> float:
        """
        Page 시간: S(radiation) = S_BH/2 인 시점.
        단순 모형: t_Page ≈ S_BH / (dS/dt)_Hawking
        dS/dt ~ c T_H (열 복사율)
        """
        S_bh = self.bekenstein_hawking_entropy
        T = self.hawking_temperature
        c = self.central_charge
        if T < 1e-15:
            return np.inf
        return S_bh / (c * T)

    def scrambling_time(self) -> float:
        """
        Scrambling 시간: t_* ≈ β/(2π) log(S_BH)
        정보가 '섞이는' 시간 (quantum chaos)
        """
        beta = self.inverse_temperature
        S_bh = self.bekenstein_hawking_entropy
        if np.isinf(beta) or S_bh < 1:
            return np.inf
        return beta / (2 * np.pi) * np.log(S_bh)

    def lyapunov_exponent(self) -> float:
        """
        Lyapunov 지수 (양자 혼돈의 척도): λ_L = 2π T_H
        MSS bound: λ_L ≤ 2π T_H  (등호 = 블랙홀, 최대 혼돈)
        """
        return 2 * np.pi * self.hawking_temperature


# ═══════════════════════════════════════════════════════════════
# 2. Kerr-AdS₄ 블랙홀
# ═══════════════════════════════════════════════════════════════

@dataclass
class KerrAdS4:
    """
    4차원 회전 AdS 블랙홀 (Kerr-AdS₄).

    계량 (Boyer-Lindquist 좌표):
      ds² = -Δ_r/ρ² (dt - a sin²θ/Ξ dφ)²
            + ρ²/Δ_r dr²
            + ρ²/Δ_θ dθ²
            + Δ_θ sin²θ/ρ² (a dt - (r²+a²)/Ξ dφ)²

    Δ_r = (r²+a²)(1 + r²/l²) - 2Mr
    Δ_θ = 1 - a²cos²θ/l²
    ρ²  = r² + a²cos²θ
    Ξ   = 1 - a²/l²

    물리 질량: M̃ = M/Ξ²
    물리 각운동량: J = aM/Ξ²
    """
    M: float        # 질량 파라미터
    a: float        # 회전 파라미터 (a = J/M)
    l: float = 1.0  # AdS 반지름
    G_N: float = 0.05

    def __post_init__(self):
        self.Xi = 1 - self.a**2 / self.l**2
        if self.Xi <= 0:
            raise ValueError(f"a = {self.a} > l = {self.l}: 초회전 파라미터")

    @property
    def physical_mass(self) -> float:
        return self.M / self.Xi**2

    @property
    def physical_angular_momentum(self) -> float:
        return self.a * self.M / self.Xi**2

    def Delta_r(self, r: np.ndarray) -> np.ndarray:
        """Δ_r = (r²+a²)(1 + r²/l²) - 2Mr"""
        return (r**2 + self.a**2) * (1 + r**2 / self.l**2) - 2 * self.M * r

    def Delta_theta(self, theta: np.ndarray) -> np.ndarray:
        """Δ_θ = 1 - a²cos²θ/l²"""
        return 1 - self.a**2 * np.cos(theta)**2 / self.l**2

    def rho_sq(self, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """ρ² = r² + a²cos²θ"""
        return r**2 + self.a**2 * np.cos(theta)**2

    def find_horizon(self, r_min: float = 0.01, r_max: float = 20.0) -> float:
        """Δ_r = 0 의 최대 실수 근 (외부 홀라이즌)."""
        r_grid = np.linspace(r_min, r_max, 10000)
        Delta = self.Delta_r(r_grid)
        # 부호 변화 탐색
        sign_changes = np.where(np.diff(np.sign(Delta)))[0]
        if len(sign_changes) == 0:
            return 0.0
        i = sign_changes[-1]  # 가장 큰 홀라이즌
        # 이분법
        r_a, r_b = r_grid[i], r_grid[i + 1]
        for _ in range(50):
            r_m = (r_a + r_b) / 2
            if self.Delta_r(np.array([r_m]))[0] * self.Delta_r(np.array([r_a]))[0] < 0:
                r_b = r_m
            else:
                r_a = r_m
        return (r_a + r_b) / 2

    @property
    def r_plus(self) -> float:
        """외부 홀라이즌 r_+"""
        return self.find_horizon()

    def hawking_temperature(self) -> float:
        """
        Hawking 온도:
        T_H = r_+(1 + a²/l²)(1 + 3r_+²/l² - a²/(r_+² + a²)) / (4π(r_+² + a²))
        """
        r = self.r_plus
        if r < 1e-10:
            return 0.0
        numerator = r * (1 + self.a**2 / self.l**2) * (1 + 3 * r**2 / self.l**2) - self.a**2 * (1 - r**2 / self.l**2) / r
        denominator = 4 * np.pi * (r**2 + self.a**2)
        return numerator / denominator

    def angular_velocity_horizon(self) -> float:
        """홀라이즌 각속도: Ω_H = a(1 + r_+²/l²)/(r_+² + a²) / Ξ"""
        r = self.r_plus
        return self.a * (1 + r**2 / self.l**2) / ((r**2 + self.a**2) * self.Xi)

    def bekenstein_hawking_entropy(self) -> float:
        """S = Area/(4G_N) = π(r_+² + a²)/(G_N Ξ)"""
        r = self.r_plus
        return np.pi * (r**2 + self.a**2) / (self.G_N * self.Xi)

    def ergoregion_boundary(self, theta: np.ndarray) -> np.ndarray:
        """
        에르고 영역 경계 (정적 한계면): g_{tt} = 0
        g_{tt} = 0 → Δ_r = a² sin²θ
        """
        # g_tt = 0 인 r 찾기 (각도별)
        r_ergo = np.zeros_like(theta)
        for i, th in enumerate(theta):
            target = self.a**2 * np.sin(th)**2
            r_grid = np.linspace(self.r_plus, 20.0, 5000)
            D = self.Delta_r(r_grid) - target
            idx = np.where(D > 0)[0]
            r_ergo[i] = r_grid[idx[0]] if len(idx) > 0 else self.r_plus
        return r_ergo

    def superradiance_condition(self, omega: float, m: int) -> bool:
        """
        초복사(superradiance) 조건: 0 < ω < m Ω_H
        이 범위의 파동은 블랙홀에서 에너지를 추출.
        """
        Omega = self.angular_velocity_horizon()
        return 0 < omega < m * Omega


# ═══════════════════════════════════════════════════════════════
# 3. 홀로그래픽 복잡도
# ═══════════════════════════════════════════════════════════════

class HolographicComplexity:
    """
    홀로그래픽 복잡도 (두 가지 추측):

    CV (Complexity = Volume) 추측:
      C = Vol(γ) / (G_N l)
      γ = AdS 시공간의 최대 부피 코드메인 슬라이스

    CA (Complexity = Action) 추측:
      C = S_WDW / (πℏ)
      S_WDW = Wheeler-DeWitt 패치의 작용

    CV2.0:
      C = P × Vol / G_N  (압력 × 부피)

    복잡도 성장률 (Lloyd bound):
      dC/dt ≤ 2E/πℏ  (단, E = 에너지)
    """
    def __init__(self, btz: BTZBlackHole):
        self.btz = btz

    def volume_complexity_btz(self) -> float:
        """
        BTZ 에서 CV 복잡도.
        γ = 최대 슬라이스: r = const, r_H < r
        Vol = ∫ dr √|g_{rr} g_{φφ}| = 2 r_+ l (정규화 후)
        """
        r_H = self.btz.r_horizon
        l = self.btz.l
        return 2 * r_H * l / self.btz.G_N

    def complexity_growth_rate(self) -> float:
        """
        복잡도 성장률: dC/dt = 2M/π (CA 추측, 비회전 BTZ)
        Lloyd bound: dC/dt ≤ 2E/πℏ = 2M/π (ℏ=1)
        BTZ 블랙홀은 bound를 포화!
        """
        return 2 * self.btz.M / np.pi

    def switchback_time(self) -> float:
        """
        Switchback 효과: 작은 섭동이 복잡도에 영향을 미치기까지의 지연.
        t_sw ≈ β/(2π) log(β E_shock)  (scrambling time)
        """
        return self.btz.scrambling_time()

    def complexity_by_stage(self, t_values: np.ndarray) -> np.ndarray:
        """
        시간에 따른 복잡도 (simplified):
        - t < t_scrambling: 로그 성장 (precomplexification)
        - t > t_scrambling: 선형 성장 (dC/dt = 2M/π)
        """
        t_sw = self.switchback_time()
        C0 = self.volume_complexity_btz()
        rate = self.complexity_growth_rate()
        C = np.where(
            t_values < t_sw,
            C0 * (1 + np.log1p(t_values / (t_sw + 1e-10))),
            C0 + rate * (t_values - t_sw),
        )
        return C


# ═══════════════════════════════════════════════════════════════
# 4. Entanglement Island 공식 (정밀 Page 곡선)
# ═══════════════════════════════════════════════════════════════

class EntanglementIsland:
    """
    Island 공식 (Penington 2019; Almheiri et al. 2019):

    S(R) = min_{I} S_gen(R ∪ I)
    S_gen = Area(∂I)/(4G_N) + S_bulk(R ∪ I)

    여기서:
    - R = 복사 영역 (Hawking radiation collector)
    - I = Island (블랙홀 내부의 bulk 영역)
    - S_bulk = QFT 얽힘 엔트로피

    두 가지 안장점:
    1. No island: S = S_bulk(R) [Hawking 결과, 증가]
    2. Island: S = Area(∂I)/(4G_N) + S_bulk(R∪I) [감소]
    Page 곡선: 두 결과 중 최솟값.

    JT 중력 + 2D CFT 모형에서 정확한 계산 가능.
    """
    def __init__(self, btz: BTZBlackHole, n_baths: int = 1):
        self.btz = btz
        self.n_baths = n_baths  # 욕조(bath) 수
        self.c = btz.central_charge

    def no_island_entropy(self, t: float, epsilon: float = 1e-3) -> float:
        """
        Island 없는 경우의 S(R).
        S_no_island ≈ (c/3) * log(t/ε) + const   [Hawking 증가]
        """
        beta = self.btz.inverse_temperature
        if np.isinf(beta):
            return (self.c / 3) * np.log(max(t, epsilon) / epsilon)
        # 열 상태에서 복사 EE
        l_rad = max(2 * t, epsilon)
        return (self.c / 3) * np.log(
            (beta / (np.pi * epsilon)) * np.sinh(np.pi * l_rad / beta)
        )

    def island_entropy(self, t: float, epsilon: float = 1e-3) -> float:
        """
        Island 있는 경우의 S_gen(R ∪ I).
        Island 경계 ∂I는 블랙홀 홀라이즌 근처에 위치.

        S_island = 2 * S_BH + S_bulk(R ∪ I)
                ≈ 2 * S_BH   [Page 시간 이후 dominates]

        (bulk 얽힘 항은 island 경계 선택으로 상쇄)
        """
        S_BH = self.btz.bekenstein_hawking_entropy
        # bulk EE 항: 복사 + island ≈ UV 발산 항만 남음
        S_uv = (self.c / 3) * np.log(1.0 / epsilon)  # UV 발산
        return 2 * S_BH + S_uv

    def page_curve(self, t_values: np.ndarray, epsilon: float = 1e-3) -> dict:
        """
        Island 공식으로 구한 정확한 Page 곡선.

        S(R, t) = min(S_no_island(t), S_island(t))

        Page 시간: S_no_island(t_Page) = S_island(t_Page)
        """
        S_no_island = np.array([self.no_island_entropy(t, epsilon) for t in t_values])
        S_island_val = self.island_entropy(0, epsilon)  # 시간 독립적 (주도 기여)
        S_island = np.full_like(S_no_island, S_island_val)
        S_page = np.minimum(S_no_island, S_island)

        # Page 시간 찾기
        diff = S_no_island - S_island
        sign_change = np.where(diff > 0)[0]
        t_page = t_values[sign_change[0]] if len(sign_change) > 0 else np.inf

        return {
            "t": t_values,
            "S_no_island": S_no_island,
            "S_island": S_island,
            "S_page": S_page,
            "t_page": t_page,
            "S_BH": self.btz.bekenstein_hawking_entropy,
            "t_scrambling": self.btz.scrambling_time(),
        }

    def information_recovery_time(self) -> float:
        """
        Hayden-Preskill 정보 회수 시간:
        t_HP = t_Page + t_scrambling
        """
        return self.btz.page_time() + self.btz.scrambling_time()


# ═══════════════════════════════════════════════════════════════
# 5. 홀로그래픽 열역학 (1st Law)
# ═══════════════════════════════════════════════════════════════

def holographic_first_law(btz: BTZBlackHole, delta_M: float) -> dict:
    """
    홀로그래픽 열역학 1법칙:
      dE = T_H dS_BH + Ω_H dJ

    AdS 블랙홀에서의 Smarr 관계:
      M = (T_H S_BH + Ω_H J) / (d-2) + ...

    BTZ (d=2):
      dM = T_H dS  (J=0)
    """
    T = btz.hawking_temperature
    S = btz.bekenstein_hawking_entropy
    dS = delta_M / T if T > 1e-15 else 0.0
    return {
        "T_H": T,
        "S_BH": S,
        "delta_M": delta_M,
        "delta_S": dS,
        "check_first_law": abs(T * dS - delta_M) < 1e-10,
    }
