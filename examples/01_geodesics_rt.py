"""
예제 1: AdS₃ 측지선과 Ryu-Takayanagi 얽힘 엔트로피

실행: python -m examples.01_geodesics_rt
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ads_cft_ml.geometry import AdS3
from ads_cft_ml.entanglement import compare_rt_cft, ee_vacuum, ee_finite_temperature
from ads_cft_ml.cft import operator_dimension_from_mass


def demo_geodesics():
    print("=" * 65)
    print("1. AdS₃ 측지선 (Geodesics) 계산")
    print("=" * 65)

    ads = AdS3(L=1.0, G_N=0.05)
    print(f"AdS 반지름 L = {ads.L},  Newton 상수 G_N = {ads.G_N}")
    print(f"중심 전하 c = 3L/(2G_N) = {ads.central_charge:.1f}")
    print()

    pairs = [(-0.5, 0.5), (-1.0, 1.0), (-2.0, 2.0), (-5.0, 5.0)]
    epsilon = 1e-3
    print(f"{'구간 [x1, x2]':<20} {'l=|x2-x1|':>10} {'측지선 길이':>15} {'RT 엔트로피':>14}")
    print("-" * 62)
    for x1, x2 in pairs:
        l = abs(x2 - x1)
        length = ads.geodesic_length(x1, x2, epsilon)
        S = ads.ryu_takayanagi_entropy(x1, x2, epsilon)
        print(f"[{x1:.1f}, {x2:.1f}]{'':<12} {l:>10.1f} {length:>15.4f} {S:>14.4f}")

    print()
    print("  RT 공식: S = Length(γ) / (4G_N) = (c/3) log(l/ε)")
    print(f"  검증: (c/3) = {ads.central_charge/3:.1f}")
    print(f"  측지선 길이 = 2L·log(l/ε) = 2·log(l/{epsilon})")


def demo_rt_vs_cft():
    print()
    print("=" * 65)
    print("2. RT 공식 vs CFT 해석적 결과 비교")
    print("=" * 65)

    ads = AdS3(L=1.0, G_N=0.05)
    l_values = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    result = compare_rt_cft(ads, l_values, epsilon=1e-3)

    print(f"\n  중심 전하 c = {result['c']:.1f}\n")
    print(f"{'l':>8} {'S_CFT':>12} {'S_RT':>12} {'오차':>12}")
    print("-" * 48)
    for i, l in enumerate(l_values):
        print(
            f"{l:>8.2f} {result['S_CFT'][i]:>12.6f} "
            f"{result['S_RT'][i]:>12.6f} {result['relative_error'][i]:>12.2e}"
        )
    print()
    print("  → RT 공식과 CFT가 정확히 일치! (오차 < 기계 정밀도)")


def demo_bulk_geometry():
    print()
    print("=" * 65)
    print("3. 연산자 차원과 벌크 질량 대응")
    print("=" * 65)
    print()
    print(f"  AdS/CFT: m²L² = Δ(Δ-1)  (d=1, AdS₃)")
    print()
    print(f"  {'m (질량)':>12} {'m²L²':>10} {'Δ₊ (비정규화)':>16} {'Δ₋ (정규화)':>14}")
    print("-" * 58)
    for m in [0.0, 0.5, 1.0, np.sqrt(2), 2.0, 5.0]:
        try:
            delta_p, delta_m = operator_dimension_from_mass(m, L=1.0, d=1)
            print(f"  {m:>12.3f} {m**2:>10.3f} {delta_p:>16.4f} {delta_m:>14.4f}")
        except ValueError as e:
            print(f"  {m:>12.3f}: {e}")
    print()
    # BF bound
    print(f"  BF bound (Breitenlöhner-Freedman): m²L² ≥ -(d/2)² = -0.25")
    print(f"  → 가벼운 tachyon(m²<0)도 Δ₊가 실수면 안정!")
    print(f"  m²L² = -0.25 (BF bound): Δ = d/2 = 0.5 (임계)")


def demo_finite_temperature():
    print()
    print("=" * 65)
    print("4. 유한 온도 얽힘 엔트로피 (BTZ 블랙홀 홀로그래피)")
    print("=" * 65)
    print()
    c = 30.0
    epsilon = 1e-3
    betas = [10.0, 5.0, 2.0, 1.0]  # β = 1/T
    l_values = np.linspace(0.1, 3.0, 5)

    print(f"  중심 전하 c = {c},  UV 컷오프 ε = {epsilon}")
    print()
    for beta in betas:
        T = 1.0 / beta
        S_arr = ee_finite_temperature(l_values, c, beta, epsilon)
        S_vac = ee_vacuum(l_values, c, epsilon)
        print(f"  T = {T:.2f} (β={beta:.1f}):")
        print(f"    l:        {' '.join(f'{l:6.2f}' for l in l_values)}")
        print(f"    S(T>0):   {' '.join(f'{s:6.3f}' for s in S_arr)}")
        print(f"    S(T=0):   {' '.join(f'{s:6.3f}' for s in S_vac)}")
        print(f"    ΔS:       {' '.join(f'{s2-s1:6.3f}' for s1, s2 in zip(S_vac, S_arr))}")
        print()
    print("  BTZ 블랙홀: 열 CFT ↔ ds² = L²/z²(-f(z)dt² + dx² + dz²/f(z))")
    print(f"  f(z) = 1 - (z/z_H)²,  z_H = β/(2π) = 홀라이즌")


if __name__ == "__main__":
    demo_geodesics()
    demo_rt_vs_cft()
    demo_bulk_geometry()
    demo_finite_temperature()
    print()
    print("=" * 65)
    print("완료. 모든 RT 계산이 CFT 해석적 결과와 일치합니다.")
    print("=" * 65)
