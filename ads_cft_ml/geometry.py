"""
AdS₃ 기하학 - Poincaré patch

계량: ds² = L²/z² * (dz² + dx² - dt²)
경계: z → 0 (UV, 고에너지)
깊은 벌크: z → ∞ (IR, 저에너지)

좌표: (t, x, z), z > 0
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class AdS3:
    """Anti-de Sitter 시공간 (2+1차원), Poincaré patch."""
    L: float = 1.0    # AdS 반지름
    G_N: float = 0.05 # Newton 상수 (c = 3L/2G_N = 30이 되도록)

    @property
    def central_charge(self) -> float:
        """Brown-Henneaux 공식: c = 3L/(2G_N)"""
        return 3 * self.L / (2 * self.G_N)

    def metric(self, z: float) -> np.ndarray:
        """
        Poincaré patch 계량 텐서 g_μν = (L/z)² diag(-1, 1, 1).
        반환: 3x3 대각 행렬 [g_tt, g_xx, g_zz]
        """
        factor = (self.L / z) ** 2
        return np.diag([-factor, factor, factor])

    def ricci_scalar(self) -> float:
        """Ricci 스칼라: R = -d(d-1)/L² = -2/L² (AdS₃)"""
        return -2.0 / self.L ** 2

    def geodesic_length(self, x1: float, x2: float, epsilon: float = 1e-3) -> float:
        """
        경계 점 (x₁, z=ε)와 (x₂, z=ε)를 잇는 최소 측지선 길이.

        측지선은 중심 x_mid = (x₁+x₂)/2, 반지름 R = |x₂-x₁|/2 인 반원.
        정규화된 길이: Δs = 2L * log(|x₂-x₁|/ε)

        유도: 계량 적분 ∫₀^π L/z * √(dx/dθ)² + (dz/dθ)² dθ
        """
        l = abs(x2 - x1)
        if l < 1e-15:
            return 0.0
        return 2 * self.L * np.log(l / epsilon)

    def geodesic_arc(
        self, x1: float, x2: float, n_points: int = 200
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        측지선 궤적 (반원). z > 0 벌크 안으로.
        반환: (x_coords, z_coords)
        """
        x_mid = (x1 + x2) / 2.0
        R = abs(x2 - x1) / 2.0
        theta = np.linspace(1e-4 * np.pi, (1 - 1e-4) * np.pi, n_points)
        x = x_mid + R * np.cos(theta)
        z = R * np.sin(theta)
        return x, z

    def ryu_takayanagi_entropy(self, x1: float, x2: float, epsilon: float = 1e-3) -> float:
        """
        Ryu-Takayanagi 공식: S = Length(γ_A) / (4G_N)

        γ_A = A 영역에 homologous한 최소 측지선
        AdS₃/CFT₂에서: S = (c/3) * log(l/ε)
        """
        length = self.geodesic_length(x1, x2, epsilon)
        return length / (4 * self.G_N)

    def turning_point(self, x1: float, x2: float) -> float:
        """측지선이 가장 깊이 들어가는 z 좌표 (= 반원 반지름)."""
        return abs(x2 - x1) / 2.0

    def bulk_to_boundary_propagator(
        self, z: float, x: float, x_prime: float, delta: float
    ) -> float:
        """
        벌크-경계 전파자 K(z, x; x').

        K(z, x; x') = (z / (z² + (x-x')²))^Δ * C_Δ
        Δ = 운영자 차원, C_Δ = 정규화 상수

        경계 극한: z→0 이면 K → δ(x - x')
        """
        C_delta = (2 * delta - 1) / np.pi  # d=1 경우 정규화
        denom = z ** 2 + (x - x_prime) ** 2
        return C_delta * (z / denom) ** delta

    def bulk_field_profile(
        self,
        z: float,
        x_grid: np.ndarray,
        source: np.ndarray,
        delta: float,
        x_prime_grid: np.ndarray,
    ) -> np.ndarray:
        """
        경계 소스 J(x')로부터 벌크 장 φ(z, x) 계산.
        φ(z, x) = ∫ dx' K(z, x; x') J(x')
        """
        phi = np.zeros_like(x_grid, dtype=float)
        dx = x_prime_grid[1] - x_prime_grid[0]
        for i, xi in enumerate(x_grid):
            K = self.bulk_to_boundary_propagator(z, xi, x_prime_grid, delta)
            phi[i] = np.trapz(K * source, dx=dx)
        return phi


def poincare_to_global(r: float, theta: float, L: float = 1.0) -> Tuple[float, float]:
    """
    Poincaré 좌표 (z, x) → 전역 AdS₃ 좌표 (ρ, φ) 변환.
    Poincaré: z = L²/(r cos θ + L), x = L r sin θ / (r cos θ + L)
    """
    rho = np.arccosh(np.sqrt(1 + r**2 / L**2))
    phi = theta
    return rho, phi
