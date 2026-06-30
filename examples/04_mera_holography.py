"""
예제 4: MERA 텐서 네트워크와 AdS/CFT 기하학

Swingle (2012): MERA의 구조가 AdS 시공간과 동형

핵심 대응:
  MERA 층 k ↔ AdS z = z_UV × 2^k
  인과 원뿔 경계 ↔ Ryu-Takayanagi 측지선
  disentangler 텐서 ↔ 측지선이 절단하는 벌크 bond
  MERA EE = O(log l) ↔ RT: S = (c/3) log(l/ε)

이 예제:
  - MERA의 얽힘 엔트로피와 RT 공식 비교
  - 층별 AdS 좌표 매핑
  - 상호 정보량과 인과 원뿔
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ads_cft_ml.tensor_network import MERA, print_mera_ads_dictionary
from ads_cft_ml.geometry import AdS3
from ads_cft_ml.entanglement import ee_vacuum


def demo_mera_structure():
    print("=" * 65)
    print("1. MERA 구조와 AdS/CFT 사전")
    print("=" * 65)

    mera = MERA(n_sites=64, L_ads=1.0, G_N=0.05)
    print()
    print_mera_ads_dictionary(mera)


def demo_mera_ee_vs_rt():
    print()
    print("=" * 65)
    print("2. MERA 얽힘 엔트로피 vs RT 공식")
    print("=" * 65)

    n_sites = 128
    mera = MERA(n_sites=n_sites, L_ads=1.0, G_N=0.05)
    ads = AdS3(L=1.0, G_N=0.05)
    c = mera.c
    epsilon_sites = 1
    epsilon_phys = 2.0 / n_sites  # 물리적 UV 컷오프 (x ∈ [-1, 1])

    print(f"\n  n_sites = {n_sites},  c = {c:.1f}")
    print(f"  물리적 격자 간격 = {2/n_sites:.4f}")
    print()
    print(f"  {'사이트 수 l':>12} {'S_MERA':>12} {'S_CFT':>12} {'S_RT':>12}")
    print("-" * 52)

    for l_sites in [2, 4, 8, 16, 32, 64]:
        l_phys = l_sites * 2.0 / n_sites  # 물리적 구간 길이
        S_mera = mera.entanglement_entropy_mera(l_sites, epsilon_sites)
        S_cft = ee_vacuum(l_phys, c, epsilon_phys)
        S_rt = ads.ryu_takayanagi_entropy(-l_phys / 2, l_phys / 2, epsilon_phys)
        print(f"  {l_sites:>12} {S_mera:>12.4f} {S_cft:>12.4f} {S_rt:>12.4f}")

    print()
    print("  MERA 얽힘 엔트로피: S ∝ log(l)")
    print("  RT 공식: S = (c/3) log(l/ε)")
    print("  → 같은 로그 스케일링! MERA ↔ AdS 기하학 확인")


def demo_causal_cone():
    print()
    print("=" * 65)
    print("3. 인과 원뿔과 RT 측지선의 대응")
    print("=" * 65)

    n_sites = 32
    mera = MERA(n_sites=n_sites, L_ads=1.0, G_N=0.05)

    print(f"\n  n_sites = {n_sites}, 층 수 = {mera.n_layers}")
    print(f"\n  구간 A의 인과 원뿔 너비 (각 층에서):")
    print()

    l_values = [2, 4, 8]
    for l in l_values:
        print(f"  |A| = {l:2d} 사이트:")
        for k in range(mera.n_layers + 1):
            width = mera.causal_cone_width(l, k)
            z_ads = mera.layers[k].z_ads
            bar = "█" * min(width, 40)
            print(f"    층 {k:2d} (z={z_ads:.4f}): 너비={width:3d} |{bar}")
        print()

    print("  → 층이 깊어질수록 인과 원뿔이 넓어짐")
    print("  → 측지선이 더 깊은 벌크로 들어가는 것과 동일")


def demo_mera_mutual_information():
    print()
    print("=" * 65)
    print("4. MERA 상호 정보량과 홀로그래픽 위상 전이")
    print("=" * 65)

    n_sites = 64
    mera = MERA(n_sites=n_sites, L_ads=1.0, G_N=0.05)
    ads = AdS3(L=1.0, G_N=0.05)
    epsilon = 2.0 / n_sites

    l_A = 8  # 각 구간 = 8 사이트
    print(f"\n  |A| = |B| = {l_A} 사이트")
    print(f"\n  gap (사이트) {'I_MERA':>12} {'I_RT':>12}")
    print("-" * 38)

    for gap in range(1, 20):
        A_start, A_end = 0, l_A
        B_start, B_end = l_A + gap, 2 * l_A + gap
        I_mera = mera.mutual_information_mera(A_start, A_end, B_start, B_end)

        # RT 상호 정보량
        l_phys = l_A * 2.0 / n_sites
        d_phys = gap * 2.0 / n_sites
        A_phys = (-l_phys - d_phys / 2, -d_phys / 2)
        B_phys = (d_phys / 2, l_phys + d_phys / 2)
        from ads_cft_ml.entanglement import mutual_information_holographic
        I_rt, phase = mutual_information_holographic(ads, A_phys, B_phys, epsilon)

        marker = " ← 위상 전이" if I_rt < 1e-6 and gap > 1 else ""
        print(f"  {gap:>14} {I_mera:>12.4f} {I_rt:>12.4f}{marker}")


def demo_holographic_geometry_summary():
    print()
    print("=" * 65)
    print("5. 홀로그래픽 기하학 요약")
    print("=" * 65)
    print()

    mera = MERA(n_sites=32, L_ads=1.0, G_N=0.05)
    ads = AdS3(L=1.0, G_N=0.05)

    print("  Swingle의 MERA ↔ AdS 대응 검증:")
    print()
    print(f"  [중심 전하]")
    print(f"    MERA: c = 3L/(2G_N) = {mera.c:.1f}")
    print(f"    RT:   c = 3L/(2G_N) = {ads.central_charge:.1f}")
    print()
    print(f"  [스케일링 차원]")
    print(f"    구간 길이 l → EE S ~ (c/3) log(l)")
    print(f"    MERA: S ~ (c/6)log2 × log2(l) = (c/6)log(l)")
    print(f"    (bond 차원 χ=2 가정)")
    print()
    print(f"  [기하학적 대응]")
    geometries = [
        ("MERA 층 k", f"AdS z = L×2^k/N = {ads.L:.1f}×2^k/{mera.n_sites}"),
        ("디스인탱글러", "측지선이 절단하는 벌크 선"),
        ("등척사상", "벌크-경계 전파자 K(z,x;x')"),
        ("인과 원뿔 넓이", "RT 측지선 z-깊이"),
        ("코스-그레이닝", "RG 흐름 (UV → IR)"),
    ]
    for mera_obj, ads_obj in geometries:
        print(f"    {mera_obj:<20} ↔  {ads_obj}")

    print()
    print("  결론: MERA는 AdS 시공간의 이산화된 표현!")
    print("        텐서 네트워크의 기하학 = 홀로그래픽 시공간")


if __name__ == "__main__":
    demo_mera_structure()
    demo_mera_ee_vs_rt()
    demo_causal_cone()
    demo_mera_mutual_information()
    demo_holographic_geometry_summary()
    print()
    print("=" * 65)
    print("완료. MERA ↔ AdS/CFT 기하학 대응 확인.")
    print("=" * 65)
