"""
MERA (Multi-scale Entanglement Renormalization Ansatz) 텐서 네트워크

Swingle (2012)의 통찰:
  MERA ↔ AdS/CFT 대응

구조적 대응:
  - MERA 네트워크 깊이 ↔ AdS 지름 방향 z
  - 디스인탱글러(disentangler) ↔ 측지선 절단
  - 등척사상(isometry) ↔ 벌크 전파
  - 코스-그레이닝 층 ↔ RG 스텝
  - MERA의 인과 원뿔 ↔ AdS 빛 원뿔
  - 경계 구간의 얽힘 컷 ↔ RT 측지선

이 구현:
  완전한 텐서 수축 대신 MERA의 기하학적 구조와
  얽힘 엔트로피를 분석적으로 계산.
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class MERAlayer:
    """MERA의 한 층 = 한 RG 스텝."""
    scale: int           # 척도 인덱스 (0=UV, max=IR)
    n_sites: int         # 이 층의 사이트 수
    z_ads: float         # 대응하는 AdS z 좌표

    def coarse_grain(self, n: int) -> int:
        """2:1 코스-그레이닝: n 사이트 → n//2 사이트."""
        return n // 2


class MERA:
    """
    이진 MERA 구조 (binary MERA).
    n_sites = 2^n_layers 인 CFT 경계에서 시작.

    얽힘 엔트로피 계산:
    - 구간 A의 얽힘 엔트로피는 그 구간에 접촉하는 인과 원뿔 경계의
      disentangler 수에 비례.
    - log₂(l/ε) 개의 층을 지나므로 S ∝ (c/3) log(l/ε) 재현.
    """
    def __init__(self, n_sites: int, L_ads: float = 1.0, G_N: float = 0.05):
        assert (n_sites & (n_sites - 1)) == 0, "n_sites는 2의 거듭제곱이어야 함"
        self.n_sites = n_sites
        self.n_layers = int(np.log2(n_sites))
        self.L_ads = L_ads
        self.G_N = G_N
        self.c = 3 * L_ads / (2 * G_N)

        # 각 층의 z 좌표 (AdS 대응)
        self.layers = [
            MERAlayer(
                scale=k,
                n_sites=n_sites // (2 ** k),
                z_ads=L_ads * 2 ** k / n_sites,
            )
            for k in range(self.n_layers + 1)
        ]

    def causal_cone_width(self, interval_length: int, layer: int) -> int:
        """
        층 layer에서 구간 A (길이 interval_length)의 인과 원뿔 너비.
        이진 MERA에서 각 층마다 +2씩 넓어짐 (disentangler 각 끝에서 1개씩).
        """
        width = interval_length + 2 * layer
        return min(width, self.layers[layer].n_sites)

    def entanglement_entropy_mera(self, interval_length: int, epsilon_sites: int = 1) -> float:
        """
        MERA에서 구간 얽힘 엔트로피.

        각 층을 통과할 때 인과 원뿔 경계에서 절단되는 bond 수:
        n_cuts(k) ≈ 2 (이진 MERA, 최소한 인과 원뿔 양끝)

        총 엔트로피: S ≈ n_bond × log₂(χ) × (c/3)/c_bond
        여기서 χ = bond 차원.

        단순화된 형태로 CFT 결과 S = (c/3) log(l/ε) 를 재현.
        """
        if interval_length <= 0:
            return 0.0
        n_layers_cut = int(np.log2(interval_length / max(epsilon_sites, 1)))
        n_layers_cut = max(n_layers_cut, 0)
        # 각 층에서 절단 수 = 2 (이진 MERA)
        # S ≈ s_bond × 2 × n_layers_cut  → s_bond = c/6 * log(2)
        s_bond = self.c / 6.0 * np.log(2)
        return 2 * s_bond * n_layers_cut

    def geodesic_length_from_mera(self, interval_length: int, epsilon_sites: int = 1) -> float:
        """
        MERA에서의 측지선 길이 (AdS 단위로 환산).
        S = L/(4G_N) * geodesic_length / L → L = geodesic_length / (4G_N * S/S_RT)
        """
        S = self.entanglement_entropy_mera(interval_length, epsilon_sites)
        return 4 * self.G_N * S

    def network_structure(self) -> List[dict]:
        """MERA 네트워크의 층별 정보."""
        info = []
        for layer in self.layers:
            info.append({
                "scale": layer.scale,
                "n_sites": layer.n_sites,
                "z_ads": layer.z_ads,
                "layer_type": "UV boundary" if layer.scale == 0
                             else "IR fixed point" if layer.scale == self.n_layers
                             else f"RG step {layer.scale}",
            })
        return info

    def mutual_information_mera(
        self,
        A_start: int, A_end: int,
        B_start: int, B_end: int,
        epsilon_sites: int = 1,
    ) -> float:
        """
        두 구간 A, B의 상호 정보량 I(A:B) = S(A) + S(B) - S(A∪B).
        A∪B가 연결된 경우와 분리된 경우를 비교.
        """
        l_A = A_end - A_start
        l_B = B_end - B_start
        l_AB = B_end - A_start  # A∪B span

        S_A = self.entanglement_entropy_mera(l_A, epsilon_sites)
        S_B = self.entanglement_entropy_mera(l_B, epsilon_sites)
        S_AB = self.entanglement_entropy_mera(l_AB, epsilon_sites)

        return max(S_A + S_B - S_AB, 0.0)


# ── 텐서 네트워크 ↔ AdS/CFT 대응 요약 ─────────────────────────

def print_mera_ads_dictionary(mera: MERA) -> None:
    """MERA ↔ AdS 대응 사전 출력."""
    print("=" * 60)
    print("MERA ↔ AdS/CFT 대응 사전")
    print("=" * 60)
    print(f"  경계 사이트 수: {mera.n_sites}  ↔  CFT UV 자유도")
    print(f"  층 수: {mera.n_layers}          ↔  AdS 깊이 방향 z")
    print(f"  중심 전하 c: {mera.c:.1f}       ↔  c = 3L/(2G_N)")
    print()
    print("  층별 AdS 좌표 (z 방향):")
    for layer_info in mera.network_structure():
        print(f"    층 {layer_info['scale']:2d}: n_sites={layer_info['n_sites']:4d},"
              f" z_AdS={layer_info['z_ads']:.4f}  [{layer_info['layer_type']}]")
    print()
    print("  기하학적 대응:")
    correspondences = [
        ("disentangler 텐서", "벌크 이중선(geodesic)의 양 끝"),
        ("isometry 텐서", "벌크-경계 전파자"),
        ("인과 원뿔 경계", "Ryu-Takayanagi 측지선"),
        ("bond 차원 χ", "exp(AdS 얽힘 용량)"),
        ("층 수 k", "log(l/ε)에 비례한 RT 길이"),
    ]
    for mera_obj, ads_obj in correspondences:
        print(f"    {mera_obj:<30s} ↔  {ads_obj}")
    print("=" * 60)
