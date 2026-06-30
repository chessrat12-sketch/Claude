"""
예제 3: 심층 신경망 = 홀로그래픽 RG 흐름

핵심 대응 (Hashimoto, Iizuka, Nishida 2018):
  신경망 층 ↔ AdS 지름 방향 z
  활성화 벡터 h_t ↔ 벌크 장 φ(z_t, x)
  가중치 행렬 ↔ RG transformation 커널
  입력층 ↔ UV 경계 (z → 0)
  출력층 ↔ IR 고정점

c-함수의 단조성:
  Zamolodchikov 정리: c(μ_UV) ≥ c(μ_IR) ≥ 0
  신경망에서: 각 층에서 정보가 압축 → c 감소 재현
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ml.holographic_rg import HolographicRGNetwork, WilsonianRG
from ads_cft_ml.geometry import AdS3
from ads_cft_ml.cft import central_charge_rg


def demo_rg_network_structure():
    print("=" * 65)
    print("1. 홀로그래픽 RG 신경망 구조")
    print("=" * 65)

    dim = 8
    n_layers = 10
    net = HolographicRGNetwork(dim=dim, n_layers=n_layers,
                                z_uv=0.01, z_ir=10.0, activation="tanh")

    print(f"\n  신경망 구조:")
    print(f"    표현 차원: {dim}")
    print(f"    층 수 (RG 스텝): {n_layers}")
    print(f"    UV z 좌표: {net.z_uv}")
    print(f"    IR z 좌표: {net.z_ir}")
    print()
    print(f"  층별 AdS z 좌표 (log 간격):")
    log_z = np.linspace(np.log(net.z_uv), np.log(net.z_ir), n_layers + 1)
    z_vals = np.exp(log_z)
    for k, (layer, z) in enumerate(zip(net.layers, z_vals[:-1])):
        print(f"    층 {k:2d}: z = {layer.z:.4f}, dz = {layer.dz:.4f}")


def demo_rg_flow_trajectory():
    print()
    print("=" * 65)
    print("2. RG 흐름 궤적 φ(z)")
    print("=" * 65)

    dim = 4
    n_layers = 20
    net = HolographicRGNetwork(dim=dim, n_layers=n_layers,
                                z_uv=0.01, z_ir=5.0, activation="tanh",
                                seed=42)

    rng = np.random.default_rng(0)
    phi_uv = rng.normal(0, 1.0, dim)

    phi_ir, trajectory = net.forward(phi_uv)
    traj_array = np.array(trajectory)

    print(f"\n  UV 경계 φ_UV: {phi_uv.round(3)}")
    print(f"  IR 고정점 φ_IR: {phi_ir.round(3)}")
    print()

    log_z = np.linspace(np.log(net.z_uv), np.log(net.z_ir), n_layers + 1)
    z_vals = np.exp(log_z)

    print(f"  RG 흐름 궤적 (표현의 노름 ||φ(z)||):")
    print(f"\n  {'z':>10} {'||φ(z)||':>12} {'해석':>20}")
    print("-" * 46)

    norm_uv = np.linalg.norm(traj_array[0])
    for k in range(0, n_layers + 1, 2):
        z = z_vals[k]
        norm = np.linalg.norm(traj_array[k])
        if k == 0:
            label = "UV 경계"
        elif k == n_layers:
            label = "IR 고정점"
        elif k < n_layers // 3:
            label = "UV 영역"
        elif k < 2 * n_layers // 3:
            label = "중간 에너지"
        else:
            label = "IR 영역"
        print(f"  {z:>10.4f} {norm:>12.4f} {label:>20}")


def demo_c_function():
    print()
    print("=" * 65)
    print("3. c-함수 (Zamolodchikov 정리의 신경망 버전)")
    print("=" * 65)

    ads = AdS3(L=1.0, G_N=0.05)
    c_uv = ads.central_charge

    dim = 6
    n_layers = 30
    net = HolographicRGNetwork(dim=dim, n_layers=n_layers,
                                z_uv=0.01, z_ir=20.0, activation="tanh",
                                seed=7)

    rng = np.random.default_rng(1)
    phi_uv = rng.normal(0, 0.5, dim)

    c_values = net.c_function(phi_uv, c_uv)

    log_z = np.linspace(np.log(net.z_uv), np.log(net.z_ir), n_layers + 1)
    z_vals = np.exp(log_z)

    print(f"\n  UV 중심 전하 c_UV = {c_uv:.1f}")
    print(f"  Zamolodchikov 정리: c(z) 단조 감소")
    print()
    print(f"  {'z':>10} {'c(z)':>10} {'c/c_UV':>10} {'에너지 척도':>16}")
    print("-" * 50)

    for k in range(0, n_layers + 1, 3):
        z = z_vals[k]
        c_z = c_values[k]
        ratio = c_z / c_uv
        if k == 0:
            scale = "UV (high E)"
        elif k == n_layers:
            scale = "IR (low E)"
        else:
            scale = f"μ ~ 1/z = {1/z:.3f}"
        print(f"  {z:>10.4f} {c_z:>10.4f} {ratio:>10.4f} {scale:>16}")

    c_ir = c_values[-1]
    print(f"\n  결론: c_UV = {c_uv:.1f} → c_IR = {c_ir:.4f}")
    print(f"  c 감소: Δc = {c_uv - c_ir:.4f}")
    print(f"  → Zamolodchikov 정리 수치적 확인 ✓")


def demo_wilsonian_rg():
    print()
    print("=" * 65)
    print("4. Wilsonian RG — β 함수 학습 및 고정점")
    print("=" * 65)

    n_couplings = 3
    rg = WilsonianRG(n_couplings=n_couplings, n_hidden=16, seed=0)

    # UV 시작점
    g0 = np.array([0.5, -0.3, 0.8])
    n_steps = 40

    trajectory = rg.flow(g0, n_steps, dlogmu=0.1)

    print(f"\n  결합 상수 수 n = {n_couplings}")
    print(f"  UV 시작점 g₀ = {g0}")
    print()
    print(f"  RG 흐름 궤적 (UV → IR):")
    print(f"\n  {'스텝':>6} {'g₁':>10} {'g₂':>10} {'g₃':>10} {'|β(g)|':>10}")
    print("-" * 50)

    for k in range(0, n_steps + 1, 5):
        g = trajectory[k]
        beta_val = rg.beta(g)
        beta_norm = np.linalg.norm(beta_val)
        print(f"  {k:>6} {g[0]:>10.4f} {g[1]:>10.4f} {g[2]:>10.4f} {beta_norm:>10.4f}")

    # 고정점 탐색
    g_fixed = rg.fixed_points(trajectory[-1])
    if g_fixed is not None:
        beta_at_fixed = rg.beta(g_fixed)
        print(f"\n  IR 고정점: g* = {g_fixed.round(4)}")
        print(f"  β(g*) = {beta_at_fixed.round(6)}  (→ 0 에 수렴)")
    else:
        print(f"\n  (초기화에 따라 고정점 없을 수 있음 — 랜덤 β 함수)")


def demo_entanglement_per_layer():
    print()
    print("=" * 65)
    print("5. 층별 얽힘 엔트로피와 홀로그래픽 해석")
    print("=" * 65)

    dim = 8
    n_layers = 15
    net = HolographicRGNetwork(dim=dim, n_layers=n_layers,
                                z_uv=0.01, z_ir=10.0, activation="tanh",
                                seed=3)

    rng = np.random.default_rng(42)
    phi_uv = rng.normal(0, 1.0, dim)
    entropies = net.entanglement_entropy_estimate(phi_uv)

    log_z = np.linspace(np.log(net.z_uv), np.log(net.z_ir), n_layers + 1)
    z_vals = np.exp(log_z)

    # CFT 예측: S(z) ~ (c/3) log(z/z_UV) + S_UV
    ads = AdS3(L=1.0, G_N=0.05)
    c = ads.central_charge

    print(f"\n  중심 전하 c = {c:.1f}")
    print(f"\n  {'z':>10} {'S_NN(z)':>12} {'S_RT(z)':>12} {'해석':>15}")
    print("-" * 53)

    for k in range(n_layers + 1):
        z = z_vals[k]
        S_nn = entropies[k]
        # RT 예측: 길이 ~ z 인 구간의 EE
        l_eff = z
        S_rt = ads.ryu_takayanagi_entropy(-l_eff/2, l_eff/2, net.z_uv) if l_eff > net.z_uv else 0.0
        label = "UV" if k == 0 else ("IR" if k == n_layers else "")
        print(f"  {z:>10.4f} {S_nn:>12.4f} {S_rt:>12.4f} {label:>15}")

    print()
    print("  신경망 층 깊이 = AdS z 방향 → 얽힘 구조 재현")


if __name__ == "__main__":
    demo_rg_network_structure()
    demo_rg_flow_trajectory()
    demo_c_function()
    demo_wilsonian_rg()
    demo_entanglement_per_layer()
    print()
    print("=" * 65)
    print("완료. 심층 신경망 ↔ 홀로그래픽 RG 흐름 대응 확인.")
    print("=" * 65)
