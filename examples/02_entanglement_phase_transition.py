"""
예제 2: 홀로그래픽 위상 전이 — 상호 정보량의 불연속

두 구간 A, B의 상호 정보량 I(A:B)에서 발생하는 1차 위상 전이:
  - Connected RT 면:    γ_{a₁b₂} ∪ γ_{a₂b₁} (두 측지선 교차)
  - Disconnected RT 면: γ_A ∪ γ_B  (각 구간 독립)

전이 조건 (대칭 구간 |A|=|B|=l, gap=d):
  connected → disconnected: log(l/d) = log(√5-2)/2 ≈ -0.48

이 전이는 "RT 위상 전이" 또는 "홀로그래픽 상관관계 소멸"이라 함.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ads_cft_ml.geometry import AdS3
from ads_cft_ml.entanglement import (
    mutual_information_holographic,
    phase_transition_parameter,
)


def demo_mutual_information_scan():
    print("=" * 65)
    print("1. 두 구간의 홀로그래픽 상호 정보량 I(A:B)")
    print("=" * 65)

    ads = AdS3(L=1.0, G_N=0.05)
    c = ads.central_charge
    epsilon = 1e-3

    print(f"\n  c = {c:.1f},  ε = {epsilon}")
    print(f"\n  A = [-l-d/2, -d/2],  B = [d/2, l+d/2]")
    print(f"  (구간 길이 l, 간격 d, 중심 대칭)\n")

    l = 2.0
    gaps = np.linspace(0.05, 6.0, 20)

    print(f"  l = {l:.1f},  gap d를 변화:")
    print(f"\n  {'d':>8} {'I(A:B)':>12} {'상태':>14} {'d/l':>8}")
    print("-" * 46)

    last_phase = None
    transition_d = None
    for d in gaps:
        A = (-l - d/2, -d/2)
        B = (d/2, l + d/2)
        I, phase = mutual_information_holographic(ads, A, B, epsilon)
        marker = ""
        if last_phase is not None and phase != last_phase:
            marker = " ← 위상 전이!"
            transition_d = d
        print(f"  {d:>8.3f} {I:>12.4f} {phase:>14}{marker}")
        last_phase = phase

    if transition_d:
        print(f"\n  → 위상 전이점: d ≈ {transition_d:.3f} (d/l ≈ {transition_d/l:.3f})")
        print(f"  이론값: d/l = √5 - 2 ≈ {np.sqrt(5) - 2:.3f}")


def demo_phase_diagram():
    print()
    print("=" * 65)
    print("2. 위상도: I(A:B) > 0 영역 (l, d 공간)")
    print("=" * 65)

    ads = AdS3(L=1.0, G_N=0.05)
    epsilon = 1e-3

    l_values = [0.5, 1.0, 2.0, 3.0, 5.0]
    print(f"\n  I > 0 (entangled) 인 최대 gap d*:")
    print(f"\n  {'l':>8} {'d* (임계)':>12} {'d*/l':>10} {'이론값 d*/l':>14}")
    print("-" * 48)

    theory_ratio = np.sqrt(5) - 2  # 대칭 구간 이론값

    for l in l_values:
        # d를 스캔하여 I가 0이 되는 점 찾기
        d_crit = None
        for d in np.linspace(0.01, 3 * l, 500):
            A = (-l - d/2, -d/2)
            B = (d/2, l + d/2)
            I, phase = mutual_information_holographic(ads, A, B, epsilon)
            if phase == "disconnected":
                d_crit = d
                break
        if d_crit:
            print(f"  {l:>8.2f} {d_crit:>12.4f} {d_crit/l:>10.4f} {theory_ratio:>14.4f}")
        else:
            print(f"  {l:>8.2f} {'미탐색':>12}")


def demo_holographic_information_paradox():
    print()
    print("=" * 65)
    print("3. 정보 역설과 홀로그래픽 위상 전이")
    print("=" * 65)
    print()
    print("  블랙홀 증발 (Hawking, 1974):")
    print("    - 블랙홀은 열복사 → 순수 상태 → 혼합 상태??")
    print("    - 정보 손실 역설")
    print()
    print("  홀로그래픽 해답 (Penington; Almheiri et al., 2019):")
    print("    - Page 곡선: S(radiation) = min(S_BH, S_rad_naive)")
    print("    - Entanglement Island: RT 면이 블랙홀 내부를 포함!")
    print()

    # Page 곡선 근사 시뮬레이션
    ads = AdS3(L=1.0, G_N=0.05)
    c = ads.central_charge
    epsilon = 1e-3

    n_steps = 50
    t_values = np.linspace(0.1, 10.0, n_steps)
    l_rad = t_values  # 복사 구간 크기 ~ 시간

    print("  홀로그래픽 Page 곡선 (단순 모형):")
    print(f"\n  {'시간 t':>10} {'S_naive':>12} {'S_island':>12} {'S_Page':>12} {'페이즈':>12}")
    print("-" * 60)

    S_BH_initial = c / 3.0 * np.log(5.0 / epsilon)  # 초기 블랙홀 엔트로피

    for i, (t, l) in enumerate(zip(t_values[::10], l_rad[::10])):
        # Hawking 복사 (naive): 단조 증가
        S_naive = c / 3.0 * np.log(l / epsilon) if l > epsilon else 0.0

        # Island 포함 RT 면: BH 엔트로피 감소
        S_BH_remaining = S_BH_initial * max(1 - t / t_values[-1], 0.01)

        # Page 곡선: 두 후보 중 최솟값
        S_page = min(S_naive, S_BH_remaining)
        phase = "Hawking" if S_naive < S_BH_remaining else "Island"

        print(f"  {t:>10.2f} {S_naive:>12.3f} {S_BH_remaining:>12.3f} {S_page:>12.3f} {phase:>12}")

    print()
    print("  → 'Island' 위상: RT 면이 블랙홀 내부 'island'를 포함")
    print("  → S(radiation)이 감소 → 정보 보존 (unitarity) 회복!")


def demo_entanglement_wedge():
    print()
    print("=" * 65)
    print("4. Entanglement Wedge 재건")
    print("=" * 65)
    print()
    print("  경계 영역 A의 entanglement wedge W(A):")
    print("    = A와 RT 면 γ_A 사이의 벌크 영역")
    print()
    print("  핵심 정리:")
    print("    - 경계 A에서 벌크 W(A) 안의 모든 정보를 재건 가능")
    print("    - 이것이 AdS/CFT의 '홀로그래픽 부분 재건' (quantum error correction)")
    print()

    ads = AdS3(L=1.0, G_N=0.05)
    epsilon = 1e-3

    print("  예시: A = [-l/2, l/2] 에 대한 entanglement wedge")
    print()
    print(f"  {'l':>6} {'RT 측지선 깊이 z*':>20} {'S_A':>10} {'재건 가능 깊이':>16}")
    print("-" * 56)
    for l in [0.5, 1.0, 2.0, 5.0]:
        z_star = ads.turning_point(-l/2, l/2)
        S = ads.ryu_takayanagi_entropy(-l/2, l/2, epsilon)
        print(f"  {l:>6.1f} {z_star:>20.4f} {S:>10.4f} {f'0 < z < {z_star:.3f}':>16}")
    print()
    print("  → l이 클수록 더 깊은 벌크까지 재건 가능")
    print("  → UV(경계) 정보 ↔ IR(깊은 벌크) 정보의 홀로그래픽 대응")


if __name__ == "__main__":
    demo_mutual_information_scan()
    demo_phase_diagram()
    demo_holographic_information_paradox()
    demo_entanglement_wedge()
    print()
    print("=" * 65)
    print("완료. 홀로그래픽 위상 전이 시뮬레이션 끝.")
    print("=" * 65)
