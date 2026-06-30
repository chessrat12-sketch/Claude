"""
벌크-경계 대응을 학습하는 신경망

AdS/CFT 사전 학습:
  입력: 경계 CFT 데이터 (소스 J(x), 1점 함수 <O(x)>)
  출력: 벌크 장 프로파일 φ(z, x)

이는 "역문제" (inverse problem):
  경계 데이터 → 벌크 내부 재건 (holographic reconstruction)

Extrapolate Dictionary (HKLL 공식):
  φ(z, x) = ∫ dx' K(z, x; x') O(x')

여기서 K(z, x; x') = (z / (z² + (x-x')²))^Δ
이 공식을 신경망으로 근사 / 역공학.
"""
import numpy as np
from typing import Tuple, List, Optional
from ..ads_cft_ml.geometry import AdS3


class HKLLReconstructionLayer:
    """
    HKLL (Hamilton-Kabat-Lifschytz-Lowe) 공식 구현.

    벌크 재건: φ(z, x) = ∫ dx' K_Δ(z, x; x') O(x')
    스미어링 함수: K_Δ(z, x; x') = c_Δ (z / (z² + (x-x')²))^Δ
    """
    def __init__(self, delta: float, L: float = 1.0):
        self.delta = delta
        self.L = L
        # 정규화 (1+1d CFT)
        import math
        self.c_delta = math.gamma(delta) / (np.sqrt(np.pi) * math.gamma(delta - 0.5))

    def smearing_function(self, z: float, x: np.ndarray, x_prime: np.ndarray) -> np.ndarray:
        """
        HKLL 스미어링 함수 K_Δ(z, x; x').
        shape: (len(x), len(x_prime)) 또는 broadcast 가능.
        """
        z2 = z ** 2
        dx2 = (x[:, None] - x_prime[None, :]) ** 2
        denom = z2 + dx2
        return self.c_delta * (self.L * z / denom) ** self.delta

    def reconstruct(
        self,
        operator_profile: np.ndarray,
        x_grid: np.ndarray,
        z: float,
    ) -> np.ndarray:
        """
        경계 연산자 O(x') → 벌크 장 φ(z, x).
        φ(z, x) = ∫ dx' K(z, x; x') <O(x')>
        """
        K = self.smearing_function(x_grid, x_grid, x_grid)
        dx = x_grid[1] - x_grid[0]
        return K @ operator_profile * dx

    def reconstruct_bulk_profile(
        self,
        operator_profile: np.ndarray,
        x_grid: np.ndarray,
        z_grid: np.ndarray,
    ) -> np.ndarray:
        """
        전체 벌크 프로파일 φ(z, x) 재건.
        반환 shape: (len(z_grid), len(x_grid))
        """
        dx = x_grid[1] - x_grid[0]
        phi = np.zeros((len(z_grid), len(x_grid)))
        for i, z in enumerate(z_grid):
            K = self.smearing_function(x_grid, x_grid, x_grid)
            phi[i] = K @ operator_profile * dx
        return phi


class BulkBoundaryNetwork:
    """
    벌크-경계 대응을 학습하는 완전연결 신경망.

    구조:
      층 0 (입력): 경계 데이터 φ_UV (z = z_min)
      층 1~L:       RG 흐름 / 벌크 전파
      층 L (출력):  예측 결과

    학습 목표:
      CFT 상관함수 → 벌크 장 + 홀로그래픽 엔트로피 일치
    """
    def __init__(
        self,
        n_boundary: int,
        n_bulk_z: int,
        delta: float = 2.0,
        L: float = 1.0,
        G_N: float = 0.05,
        seed: int = 0,
    ):
        self.n_boundary = n_boundary
        self.n_bulk_z = n_bulk_z
        self.delta = delta
        self.adspace = AdS3(L=L, G_N=G_N)
        self.hkll = HKLLReconstructionLayer(delta, L)

        rng = np.random.default_rng(seed)
        # 가중치: 각 z-층마다 독립적인 선형 변환
        self.weights = [
            rng.normal(0, 1.0 / np.sqrt(n_boundary), (n_boundary, n_boundary))
            for _ in range(n_bulk_z)
        ]

    def forward_hkll(
        self,
        operator_data: np.ndarray,
        x_grid: np.ndarray,
        z_grid: np.ndarray,
    ) -> np.ndarray:
        """HKLL 공식으로 벌크 재건."""
        return self.hkll.reconstruct_bulk_profile(operator_data, x_grid, z_grid)

    def one_point_function(
        self,
        bulk_profile: np.ndarray,
        z_grid: np.ndarray,
        epsilon: float = 1e-3,
    ) -> np.ndarray:
        """
        벌크 장의 정규화 모드에서 1점 함수 추출.
        <O(x)> = lim_{z→0} z^{-Δ} φ(z, x)  (비정규화 모드 제거 후)

        z → 0 극한에서 정규화 모드: φ(z) ~ z^Δ <O>
        """
        # 가장 작은 z에서 외삽
        if len(z_grid) < 2:
            return bulk_profile[0]
        z0, z1 = z_grid[0], z_grid[1]
        phi0, phi1 = bulk_profile[0], bulk_profile[1]
        # 로그 외삽: d(log φ)/d(log z) ~ Δ
        log_slope = np.log(np.abs(phi1 + 1e-15) / np.abs(phi0 + 1e-15)) / np.log(z1 / z0)
        # 정규화 모드 계수
        return phi0 / (z0 ** self.delta + 1e-15)

    def holographic_entanglement(
        self,
        bulk_profile: np.ndarray,
        x_grid: np.ndarray,
        z_grid: np.ndarray,
        A_indices: Tuple[int, int],
    ) -> float:
        """
        벌크 장 프로파일에서 홀로그래픽 얽힘 엔트로피 추출.
        RT 측지선이 통과하는 벌크 장의 값으로 엔트로피 추정.
        """
        i1, i2 = A_indices
        x1, x2 = x_grid[i1], x_grid[i2]
        # 직접 RT 공식 사용
        return self.adspace.ryu_takayanagi_entropy(x1, x2, x_grid[1] - x_grid[0])


class NeuralHolography:
    """
    AdS/CFT 사전을 모두 통합한 홀로그래피 ML 파이프라인.

    파이프라인:
      1. 경계 소스 J(x) 생성
      2. CFT 상관함수 계산 (해석적)
      3. HKLL로 벌크 재건
      4. RT 공식으로 홀로그래픽 EE 계산
      5. CFT EE와 비교 (검증)
    """
    def __init__(self, n_sites: int = 64, L: float = 1.0, G_N: float = 0.05):
        self.n_sites = n_sites
        self.L = L
        self.G_N = G_N
        self.adspace = AdS3(L=L, G_N=G_N)
        self.c = self.adspace.central_charge
        self.x_grid = np.linspace(-5, 5, n_sites)

    def generate_source(self, width: float = 0.5, center: float = 0.0) -> np.ndarray:
        """가우시안 소스 J(x) = exp(-(x-center)²/2σ²)."""
        return np.exp(-0.5 * ((self.x_grid - center) / width) ** 2)

    def cft_two_point(self, delta: float) -> np.ndarray:
        """
        소스 J(x)에 의한 1점 함수: <O(x)> ∝ ∫ dx' |x-x'|^{-2Δ} J(x').
        """
        dx = self.x_grid[1] - self.x_grid[0]
        source = self.generate_source()
        G = np.zeros((self.n_sites, self.n_sites))
        for i in range(self.n_sites):
            for j in range(self.n_sites):
                dij = abs(self.x_grid[i] - self.x_grid[j])
                G[i, j] = dij ** (-2 * delta) if dij > 1e-3 else 0.0
        return G @ source * dx

    def run_holographic_pipeline(
        self,
        l_values: np.ndarray,
        epsilon: float = 1e-2,
    ) -> dict:
        """
        전체 홀로그래픽 파이프라인 실행 및 CFT vs RT 비교.
        """
        from ..ads_cft_ml.entanglement import ee_vacuum

        S_cft = ee_vacuum(l_values, self.c, epsilon)
        S_rt = np.array([
            self.adspace.ryu_takayanagi_entropy(-l / 2, l / 2, epsilon)
            for l in l_values
        ])

        return {
            "x_grid": self.x_grid,
            "l_values": l_values,
            "S_CFT": S_cft,
            "S_RT": S_rt,
            "central_charge": self.c,
            "match": np.allclose(S_cft, S_rt, rtol=1e-10),
        }
