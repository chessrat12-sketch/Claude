"""
예제 6: PyTorch 학습 루프

세 가지 신경망 학습:
  1. BulkBoundaryNet   — CFT 소스 → 벌크 장 재건
  2. HolographicEENet  — 얽힘 엔트로피 학습 + 중심 전하 c 역공학
  3. BetaFunctionNet   — β 함수 학습 (RG 흐름 역공학)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml.training import (
    TrainConfig,
    train_bulk_boundary,
    train_ee_network,
    train_beta_function,
    BulkBoundaryNet,
    HolographicEENet,
    BetaFunctionNet,
    CFTBoundaryDataset,
    EntanglementDataset,
)
from ads_cft_ml.geometry import AdS3
from ads_cft_ml.entanglement import holographic_ee, ee_vacuum

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
os.makedirs(OUTDIR, exist_ok=True)


def demo_bulk_boundary_training():
    print("=" * 60)
    print("1. BulkBoundaryNet 학습")
    print("   CFT 소스 J(x) → 1점 함수 <O(x)>")
    print("=" * 60)

    config = TrainConfig(lr=3e-4, batch_size=32, n_epochs=20)
    print(f"\n  설정: lr={config.lr}, batch={config.batch_size}, epoch={config.n_epochs}\n")

    model, train_losses, val_losses = train_bulk_boundary(
        config=config, n_samples=800, n_sites=32, verbose=True
    )

    print(f"\n  최종 학습 손실: {train_losses[-1]:.5f}")
    print(f"  최종 검증 손실: {val_losses[-1]:.5f}")
    print(f"  최선 검증 손실: {min(val_losses):.5f}")

    # 정성적 테스트
    model.eval()
    dataset = CFTBoundaryDataset(n_samples=10, n_sites=32)
    J, O_true = dataset[0]
    with torch.no_grad():
        out = model(J.unsqueeze(0))
    O_pred = out["one_point_fn"].squeeze().numpy()
    O_true_np = O_true.numpy()
    corr = np.corrcoef(O_pred, O_true_np)[0, 1]
    print(f"\n  예측 vs 정답 상관계수: {corr:.4f}")

    # 손실 그래프
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(train_losses, label="학습", color="#E74C3C", lw=2)
    ax.plot(val_losses, label="검증", color="#3498DB", lw=2)
    ax.set_xlabel("에폭", fontsize=10)
    ax.set_ylabel("손실 (MSE)", fontsize=10)
    ax.set_title("BulkBoundaryNet 학습 곡선\nCFT 소스 → 1점 함수", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    ax = axes[1]
    x_grid = np.linspace(-5, 5, 32)
    ax.plot(x_grid, O_true_np, label="정답 <O(x)>", color="#E74C3C", lw=2.5)
    ax.plot(x_grid, O_pred, label="예측", color="#3498DB", lw=2, ls="--")
    ax.plot(x_grid, J.numpy(), label="소스 J(x)", color="gray", lw=1.5, alpha=0.7)
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("진폭", fontsize=10)
    ax.set_title(f"CFT 소스 → 1점 함수 재건 (r={corr:.4f})", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUTDIR, "09_bulk_boundary_training.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  그래프 저장: {os.path.basename(path)}")
    return model


def demo_ee_training():
    print()
    print("=" * 60)
    print("2. HolographicEENet 학습")
    print("   얽힘 엔트로피 학습 + 중심 전하 c 역공학")
    print("=" * 60)

    ads = AdS3(L=1.0, G_N=0.05)
    true_c = ads.central_charge
    print(f"\n  정답 c = {true_c:.1f}\n")

    config = TrainConfig(lr=1e-3, batch_size=64, n_epochs=30)
    model, train_losses, val_losses, learned_c = train_ee_network(
        config=config, n_samples=1500, n_intervals=12, verbose=True
    )

    print(f"\n  학습된 c = {learned_c:.4f}  (정답 {true_c:.1f})")
    print(f"  오차: {abs(learned_c - true_c) / true_c * 100:.2f}%")

    # 예측 비교
    model.eval()
    epsilon = 1e-3
    l_test = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    log_l = np.log(l_test / epsilon)
    features = np.zeros((1, 12 * 4), dtype=np.float32)
    for j in range(min(12, len(l_test))):
        l_j = l_test[j % len(l_test)]
        features[0, j*4] = l_j
        features[0, j*4+1] = l_j**2
        features[0, j*4+2] = np.log(l_j / epsilon)
        features[0, j*4+3] = 1.0

    with torch.no_grad():
        S_pred = model(torch.tensor(features)).squeeze().numpy()
    S_rt = np.array([holographic_ee(ads, l, epsilon) for l in l_test])

    print(f"\n  {'l':>8} {'S_RT':>12} {'S_pred':>12} {'오차%':>10}")
    print("-" * 46)
    for j, l in enumerate(l_test[:min(12, 5)]):
        idx = j
        if idx < len(S_pred):
            err = abs(S_pred[idx] - S_rt[j]) / abs(S_rt[j]) * 100
            print(f"  {l:>8.2f} {S_rt[j]:>12.4f} {S_pred[idx]:>12.4f} {err:>10.2f}%")

    # 학습 곡선 + c 수렴
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.semilogy(train_losses, label="학습", color="#E74C3C", lw=2)
    ax.semilogy(val_losses, label="검증", color="#3498DB", lw=2)
    ax.axhline(min(val_losses), color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("에폭", fontsize=10)
    ax.set_ylabel("손실 (MSE, log)", fontsize=10)
    ax.set_title(f"HolographicEENet 학습 곡선\n학습된 c={learned_c:.2f} (정답={true_c:.1f})", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    l_plot = np.linspace(0.05, 12, 200)
    S_cft_plot = ee_vacuum(l_plot, true_c, epsilon)
    ax.plot(l_plot, S_cft_plot, color="#E74C3C", lw=2.5, label=f"RT 공식 (c={true_c:.1f})")
    # 학습된 모델로 예측
    x_plot = np.zeros((1, 12 * 4), dtype=np.float32)
    S_model_plot = []
    for l_val in l_plot[::5]:
        for j in range(12):
            x_plot[0, j*4] = l_val
            x_plot[0, j*4+1] = l_val**2
            x_plot[0, j*4+2] = np.log(max(l_val, epsilon) / epsilon)
            x_plot[0, j*4+3] = 1.0
        with torch.no_grad():
            s = model(torch.tensor(x_plot)).squeeze()[0].item()
        S_model_plot.append(s)
    ax.plot(l_plot[::5], S_model_plot, color="#3498DB", lw=2, ls="--",
            label=f"신경망 예측 (c≈{learned_c:.1f})")
    ax.set_xlabel("구간 길이 l", fontsize=10)
    ax.set_ylabel("얽힘 엔트로피 S", fontsize=10)
    ax.set_title("RT 공식 vs 학습된 신경망", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUTDIR, "10_ee_training.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  그래프 저장: {os.path.basename(path)}")
    return model, learned_c


def demo_beta_function_training():
    print()
    print("=" * 60)
    print("3. BetaFunctionNet 학습")
    print("   RG β 함수 역공학 + 고정점 탐색")
    print("=" * 60)

    config = TrainConfig(lr=2e-3, batch_size=128, n_epochs=30)
    model, losses = train_beta_function(
        config=config, n_couplings=3, verbose=True
    )

    # 학습된 β 함수로 RG 흐름 생성
    g0 = torch.tensor([0.8, -0.5, 0.3], dtype=torch.float32)
    trajectory = model.rg_flow(g0, n_steps=80, dlogmu=0.05)
    traj_np = trajectory.numpy()

    print(f"\n  UV 시작: g = {g0.numpy().round(3)}")
    print(f"  IR 고정점 근방: g = {traj_np[-1].round(4)}")
    beta_ir = model(torch.tensor(traj_np[-1], dtype=torch.float32))
    print(f"  |β(g_IR)| = {beta_ir.norm().item():.6f}  (→ 0 이면 고정점)")

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.semilogy(losses, color="#E74C3C", lw=2)
    ax.set_xlabel("에폭", fontsize=10)
    ax.set_ylabel("손실 (log)", fontsize=10)
    ax.set_title("β 함수 학습 곡선", fontsize=10)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    steps = np.arange(len(traj_np))
    colors_g = ["#E74C3C", "#3498DB", "#2ECC71"]
    for i, (gname, color) in enumerate(zip(["g₁", "g₂", "g₃"], colors_g)):
        ax.plot(steps, traj_np[:, i], color=color, lw=2, label=gname)
    ax.set_xlabel("RG 스텝 (UV→IR)", fontsize=10)
    ax.set_ylabel("결합 상수 g", fontsize=10)
    ax.set_title("학습된 β 함수의 RG 흐름", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 3D RG 궤적
    ax3d = fig.add_subplot(1, 3, 3, projection="3d")
    ax3d.plot(traj_np[:, 0], traj_np[:, 1], traj_np[:, 2],
              color="#2C3E50", lw=2, alpha=0.8)
    ax3d.scatter(*traj_np[0], color="#E74C3C", s=80, label="UV", zorder=5)
    ax3d.scatter(*traj_np[-1], color="#3498DB", s=80, label="IR", zorder=5)
    ax3d.set_xlabel("g₁", fontsize=8)
    ax3d.set_ylabel("g₂", fontsize=8)
    ax3d.set_zlabel("g₃", fontsize=8)
    ax3d.set_title("RG 흐름 궤적\n(결합 상수 공간)", fontsize=9)
    ax3d.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUTDIR, "11_beta_function_training.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  그래프 저장: {os.path.basename(path)}")
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch 홀로그래픽 ML 학습 실험")
    print("=" * 60)

    model_bb = demo_bulk_boundary_training()
    model_ee, learned_c = demo_ee_training()
    model_beta = demo_beta_function_training()

    print()
    print("=" * 60)
    print("학습 완료 요약")
    print("=" * 60)
    print(f"  BulkBoundaryNet: CFT 소스 → 1점 함수 학습 완료")
    print(f"  HolographicEENet: 학습된 중심 전하 c ≈ {learned_c:.2f} (정답 30.0)")
    print(f"  BetaFunctionNet: RG β 함수 역공학 완료")
    print()
    print("  그래프: plots/09_*.png, 10_*.png, 11_*.png")
