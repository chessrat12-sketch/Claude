"""
예제 7: 고급 홀로그래픽 물리

1. BTZ 블랙홀 (비회전 + 회전)
2. Kerr-AdS₄ (회전 4차원 AdS 블랙홀)
3. 홀로그래픽 복잡도 (Complexity = Volume)
4. Entanglement Island 공식 (정밀 Page 곡선)
5. 양자 혼돈과 Lyapunov 지수 (MSS bound)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ads_cft_ml.advanced_physics import (
    BTZBlackHole, KerrAdS4, HolographicComplexity,
    EntanglementIsland, holographic_first_law,
)


def demo_btz():
    print("=" * 65)
    print("1. BTZ 블랙홀")
    print("=" * 65)

    print("\n  [1a] 비회전 BTZ (J=0) — 열역학")
    print(f"\n  {'M':>6} {'r_H':>8} {'T_H':>10} {'S_BH':>10} {'t_Page':>10} {'t_scr':>10}")
    print("-" * 60)
    for M in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        btz = BTZBlackHole(M=M, l=1.0, G_N=0.05)
        t_p = btz.page_time()
        t_sc = btz.scrambling_time()
        print(f"  {M:>6.1f} {btz.r_horizon:>8.4f} {btz.hawking_temperature:>10.4f} "
              f"{btz.bekenstein_hawking_entropy:>10.3f} {t_p:>10.3f} {t_sc:>10.3f}")

    print("\n  BTZ 블랙홀 = 열 CFT 상태 → 유한 온도 얽힘 엔트로피")
    btz = BTZBlackHole(M=1.0, l=1.0, G_N=0.05)
    beta = btz.inverse_temperature
    c = btz.central_charge
    epsilon = 1e-3
    print(f"\n  M=1: T_H={btz.hawking_temperature:.4f}, β={beta:.4f}, c={c:.1f}")
    print(f"\n  {'l':>8} {'S(T>0)':>12} {'S(T=0)':>12} {'열 보정 ΔS':>14}")
    print("-" * 50)
    for l in [0.5, 1.0, 2.0, 5.0]:
        S_T = btz.thermal_entanglement_entropy(l, epsilon)
        S_0 = btz.thermal_entanglement_entropy(l, epsilon)  # 진공과 비교
        from ads_cft_ml.entanglement import ee_vacuum
        S_vac = ee_vacuum(l, c, epsilon)
        print(f"  {l:>8.2f} {S_T:>12.4f} {S_vac:>12.4f} {S_T-S_vac:>14.4f}")

    print("\n  [1b] 회전 BTZ (J≠0) — r_± 구조")
    print(f"\n  M=2, l=1, G_N=0.05")
    print(f"\n  {'J':>8} {'r_+':>8} {'r_-':>8} {'T_H':>10} {'Ω_H':>10} {'S_BH':>10}")
    print("-" * 58)
    M = 2.0
    for J_frac in [0.0, 0.3, 0.6, 0.9, 0.99]:
        J = J_frac * M
        btz = BTZBlackHole(M=M, l=1.0, G_N=0.05, J=J)
        print(f"  {J:>8.3f} {btz.r_horizon:>8.4f} {btz.r_inner:>8.4f} "
              f"{btz.hawking_temperature:>10.5f} {btz.angular_velocity:>10.5f} "
              f"{btz.bekenstein_hawking_entropy:>10.3f}")

    print("\n  J → Ml: 극한 회전 (T_H → 0, 'extremal BTZ')")

    print("\n  [1c] 준정규 모드 (Quasinormal Modes)")
    btz = BTZBlackHole(M=1.0, l=1.0, G_N=0.05)
    qnm = btz.quasinormal_frequencies(n=5)
    print(f"\n  M=1, T_H={btz.hawking_temperature:.4f}")
    print(f"\n  QNM 주파수 ω_n = -i 4π T_H (n+1):  [순허수 = 과감쇠]")
    for n, omega in enumerate(qnm):
        print(f"    n={n}: ω = {omega.real:+.4f} {omega.imag:+.4f}i   "
              f"|Im(ω)| = {abs(omega.imag):.4f}")


def demo_kerr_ads4():
    print()
    print("=" * 65)
    print("2. Kerr-AdS₄ (회전하는 4차원 AdS 블랙홀)")
    print("=" * 65)

    print(f"\n  계량: ds² = -Δ_r/ρ²(dt - a sin²θ/Ξ dφ)² + ρ²/Δ_r dr² + ...")
    print(f"  Δ_r = (r²+a²)(1+r²/l²) - 2Mr")
    print()

    M = 1.0
    l = 1.0
    print(f"  M={M}, l={l}, G_N=0.05")
    print(f"\n  {'a':>8} {'r_+':>8} {'T_H':>10} {'Ω_H':>10} {'S_BH':>12} {'초복사 조건':>16}")
    print("-" * 68)

    for a in [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]:
        try:
            kerr = KerrAdS4(M=M, a=a, l=l, G_N=0.05)
            r_p = kerr.r_plus
            T = kerr.hawking_temperature()
            Omega = kerr.angular_velocity_horizon()
            S = kerr.bekenstein_hawking_entropy()
            # ω=0.3, m=1 에서 초복사 확인
            super_rad = kerr.superradiance_condition(omega=0.3, m=1)
            sr_str = "YES (0.3 < mΩ)" if super_rad else "no"
            print(f"  {a:>8.2f} {r_p:>8.4f} {T:>10.5f} {Omega:>10.5f} {S:>12.3f} {sr_str:>16}")
        except ValueError as e:
            print(f"  {a:>8.2f}: {e}")

    print()
    print("  a → l: Ξ = 1 - a²/l² → 0 (초회전 한계)")
    print("  극한 회전 (a→l): T_H → 0, S_BH 최대")
    print("  초복사: 0 < ω < mΩ_H → 블랙홀에서 에너지 추출 가능!")

    # Kerr-AdS superradiance 상세
    print(f"\n  [초복사 상세 — a=0.8, M=1, l=1]")
    kerr = KerrAdS4(M=M, a=0.8, l=l, G_N=0.05)
    Omega = kerr.angular_velocity_horizon()
    print(f"  홀라이즌 각속도 Ω_H = {Omega:.4f}")
    print(f"  초복사 조건: 0 < ω < m Ω_H")
    print(f"\n  {'m':>4} {'Ω_H × m':>10} {'ω=0.2 초복사?':>16} {'ω=0.5 초복사?':>16}")
    print("-" * 50)
    for m in [1, 2, 3]:
        sr_02 = kerr.superradiance_condition(0.2, m)
        sr_05 = kerr.superradiance_condition(0.5, m)
        print(f"  {m:>4} {Omega*m:>10.4f} {'YES' if sr_02 else 'no':>16} {'YES' if sr_05 else 'no':>16}")


def demo_holographic_complexity():
    print()
    print("=" * 65)
    print("3. 홀로그래픽 복잡도 (Complexity = Volume)")
    print("=" * 65)

    print("\n  두 추측:")
    print("    CV: C = Vol(최대 코드메인 슬라이스) / G_N l")
    print("    CA: C = S_WDW / π  (Wheeler-DeWitt 패치 작용)")
    print("  Lloyd bound: dC/dt ≤ 2E/π  (ℏ=1)")
    print("  BTZ: dC/dt = 2M/π  (bound 포화!)")
    print()

    M_vals = [0.5, 1.0, 2.0, 5.0]
    t_max = 30.0
    t_vals = np.linspace(0, t_max, 300)

    print(f"  {'M':>6} {'C(t=0)':>12} {'dC/dt':>12} {'Lloyd bound':>14} {'포화?':>8}")
    print("-" * 56)
    for M in M_vals:
        btz = BTZBlackHole(M=M, l=1.0, G_N=0.05)
        comp = HolographicComplexity(btz)
        C0 = comp.volume_complexity_btz()
        rate = comp.complexity_growth_rate()
        lloyd = 2 * M / np.pi
        sat = "YES ✓" if abs(rate - lloyd) < 1e-10 else "no"
        print(f"  {M:>6.1f} {C0:>12.3f} {rate:>12.4f} {lloyd:>14.4f} {sat:>8}")

    # 복잡도 시간 진화
    print(f"\n  M=1, 시간별 복잡도 C(t):")
    btz1 = BTZBlackHole(M=1.0, l=1.0, G_N=0.05)
    comp1 = HolographicComplexity(btz1)
    t_sample = [0, 5, 10, 20, 30]
    C_traj = comp1.complexity_by_stage(np.array(t_sample, dtype=float))
    t_sw = comp1.switchback_time()
    print(f"  Scrambling 시간 t_* = {t_sw:.3f}")
    print(f"\n  {'t':>6} {'C(t)':>12} {'단계':>20}")
    print("-" * 42)
    for t, C in zip(t_sample, C_traj):
        stage = "초기 (로그 성장)" if t < t_sw else "후기 (선형 성장)"
        print(f"  {t:>6.0f} {C:>12.3f} {stage:>20}")


def demo_island_formula():
    print()
    print("=" * 65)
    print("4. Entanglement Island 공식 — 정밀 Page 곡선")
    print("=" * 65)

    print("\n  Island 공식: S(R) = min_{I} [Area(∂I)/4G + S_bulk(R∪I)]")
    print("  → 정보 보존(unitarity) + RT 공식의 자연스러운 일반화")
    print()

    for M, label in [(0.5, "가벼운 BH"), (1.0, "중간 BH"), (3.0, "무거운 BH")]:
        btz = BTZBlackHole(M=M, l=1.0, G_N=0.05)
        island = EntanglementIsland(btz)
        t_vals = np.linspace(0.01, btz.page_time() * 2, 5)
        result = island.page_curve(t_vals, epsilon=1e-3)

        t_page = result["t_page"]
        S_BH = result["S_BH"]
        t_sc = result["t_scrambling"]
        t_HP = island.information_recovery_time()

        print(f"  {label} (M={M}):")
        print(f"    T_H = {btz.hawking_temperature:.4f}")
        print(f"    S_BH = {S_BH:.2f}  (최대 얽힘 엔트로피)")
        print(f"    t_Page = {t_page:.2f}  (얽힘 반환점)")
        print(f"    t_scrambling = {t_sc:.3f}  (양자 혼돈)")
        print(f"    t_HP = {t_HP:.2f}  (Hayden-Preskill 정보 회수)")
        print(f"    t_Page/t_scrambling = {t_page/t_sc if t_sc > 0 else 'inf':.2f}")
        print()

    print("  결론:")
    print("    Island 없음: S(복사) → ∞  [정보 손실, unitarity 위반]")
    print("    Island 포함: S(복사) ≤ 2 S_BH  [Page 곡선, unitarity 보존]")
    print("    → 홀로그래피가 블랙홀 정보 역설을 해결!")


def demo_quantum_chaos():
    print()
    print("=" * 65)
    print("5. 양자 혼돈과 MSS bound")
    print("=" * 65)

    print("\n  Maldacena-Shenker-Stanford (MSS) bound:")
    print("  λ_L ≤ 2π T  (Lyapunov 지수 상한)")
    print("  BTZ 블랙홀: λ_L = 2π T_H  (bound 포화 → 최대 혼돈)")
    print()
    print("  OTOC (Out-of-Time-Order Correlator):")
    print("  <W(t)V(0)W(t)V(0)> ~ 1 - ε e^{λ_L t}  (t < t_scrambling)")
    print()

    print(f"  {'M':>6} {'T_H':>10} {'λ_L=2πT':>12} {'t_scr β/(2π)log(S)':>22} {'t_Page':>10}")
    print("-" * 64)
    for M in [0.1, 0.5, 1.0, 2.0, 5.0]:
        btz = BTZBlackHole(M=M, l=1.0, G_N=0.05)
        lam = btz.lyapunov_exponent()
        t_sc = btz.scrambling_time()
        t_p = btz.page_time()
        print(f"  {M:>6.1f} {btz.hawking_temperature:>10.4f} {lam:>12.4f} "
              f"{t_sc:>22.4f} {t_p:>10.3f}")

    print()
    print("  t_scrambling < t_Page: 정보가 먼저 '섞인' 뒤 Page 시간에 회수")
    print("  → Hayden-Preskill 프로토콜: 작은 복사 조각으로 전체 정보 복원 가능")

    print("\n  [홀로그래픽 1법칙 검증]")
    btz = BTZBlackHole(M=1.0, l=1.0, G_N=0.05)
    result = holographic_first_law(btz, delta_M=0.01)
    print(f"  M=1, ΔM=0.01:")
    print(f"    T_H = {result['T_H']:.6f}")
    print(f"    ΔS = ΔM/T_H = {result['delta_S']:.6f}")
    print(f"    T_H × ΔS = {result['T_H'] * result['delta_S']:.8f}")
    print(f"    ΔM = {result['delta_M']:.8f}")
    print(f"    1법칙 dE=TdS 만족? {result['check_first_law']}")


if __name__ == "__main__":
    demo_btz()
    demo_kerr_ads4()
    demo_holographic_complexity()
    demo_island_formula()
    demo_quantum_chaos()

    print()
    print("=" * 65)
    print("완료. 고급 홀로그래픽 물리 계산 끝.")
    print("=" * 65)
