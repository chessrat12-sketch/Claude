"""
홀로그래픽 물리 시각화 (matplotlib)

생성되는 그래프:
  1. AdS₃ 측지선 (Poincaré 반평면)
  2. RT vs CFT 얽힘 엔트로피 비교
  3. 홀로그래픽 위상 전이 (상호 정보량)
  4. Page 곡선 (Island 공식)
  5. MERA 텐서 네트워크 다이어그램
  6. 홀로그래픽 RG 흐름
  7. 벌크 장 프로파일 φ(z,x)
  8. BTZ 블랙홀 열역학
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 헤드리스 환경
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from typing import Optional, List
import os

from .geometry import AdS3
from .entanglement import (
    ee_vacuum, ee_finite_temperature, holographic_ee,
    mutual_information_holographic,
)
from .advanced_physics import BTZBlackHole, EntanglementIsland


OUTDIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
os.makedirs(OUTDIR, exist_ok=True)

PALETTE = {
    "uv": "#E74C3C",
    "ir": "#3498DB",
    "mid": "#2ECC71",
    "accent": "#F39C12",
    "dark": "#2C3E50",
    "light": "#ECF0F1",
}


def save(fig: plt.Figure, name: str) -> str:
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ── 1. AdS₃ 측지선 ────────────────────────────────────────────

def plot_geodesics(save_path: Optional[str] = None) -> str:
    ads = AdS3(L=1.0, G_N=0.05)
    fig, ax = plt.subplots(figsize=(9, 6))

    # 경계 (z = 0)
    ax.axhline(0, color=PALETTE["dark"], lw=2, label="AdS 경계 (CFT₂)")
    ax.fill_between([-4, 4], 0, 5, alpha=0.05, color=PALETTE["ir"])

    # 여러 구간의 측지선
    intervals = [
        (-0.5, 0.5, PALETTE["uv"], "l=1"),
        (-1.0, 1.0, PALETTE["accent"], "l=2"),
        (-2.0, 2.0, PALETTE["mid"], "l=4"),
        (-3.0, 3.0, PALETTE["ir"], "l=6"),
    ]
    for x1, x2, color, label in intervals:
        x, z = ads.geodesic_arc(x1, x2)
        ax.plot(x, z, color=color, lw=2.5, label=f"RT 측지선 {label}")
        ax.scatter([x1, x2], [0, 0], color=color, s=50, zorder=5)

    # AdS 격자선 (등z 선)
    for z_val in [0.5, 1.0, 1.5, 2.0, 2.5]:
        ax.axhline(z_val, color="gray", lw=0.5, alpha=0.3, ls="--")
        ax.text(3.7, z_val + 0.05, f"z={z_val}", fontsize=7, color="gray")

    ax.set_xlim(-4, 4)
    ax.set_ylim(-0.1, 3.2)
    ax.set_xlabel("경계 좌표 x", fontsize=12)
    ax.set_ylabel("벌크 방향 z (UV→IR)", fontsize=12)
    ax.set_title("AdS₃ Poincaré 반평면: 홀로그래픽 측지선\n"
                 r"$S = \mathrm{Length}(\gamma_A)/(4G_N) = \frac{c}{3}\log(l/\varepsilon)$",
                 fontsize=13)
    ax.legend(loc="upper right", fontsize=9)

    # 화살표: UV → IR
    ax.annotate("", xy=(3.8, 3.0), xytext=(3.8, 0.1),
                arrowprops=dict(arrowstyle="->", color=PALETTE["dark"], lw=1.5))
    ax.text(3.82, 1.5, "z↑\n(IR)", fontsize=9, color=PALETTE["dark"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = save_path or save(fig, "01_geodesics.png")
    return path


# ── 2. RT vs CFT 얽힘 엔트로피 ────────────────────────────────

def plot_ee_comparison(save_path: Optional[str] = None) -> str:
    ads = AdS3(L=1.0, G_N=0.05)
    c = ads.central_charge
    epsilon = 1e-3
    l = np.linspace(0.01, 20, 300)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 왼쪽: 진공 vs 유한 온도
    ax = axes[0]
    S_vac = ee_vacuum(l, c, epsilon)
    ax.plot(l, S_vac, color=PALETTE["dark"], lw=2.5, label="T=0 진공")

    for beta, color, ls in [(10, PALETTE["mid"], "--"),
                             (2, PALETTE["accent"], "-."),
                             (0.5, PALETTE["uv"], ":")]:
        S_T = ee_finite_temperature(l, c, beta, epsilon)
        ax.plot(l, S_T, color=color, lw=2, ls=ls,
                label=f"T={1/beta:.1f} (BTZ 블랙홀)")

    ax.set_xlabel("구간 길이 l", fontsize=11)
    ax.set_ylabel("얽힘 엔트로피 S", fontsize=11)
    ax.set_title(f"Calabrese-Cardy vs BTZ 홀로그래픽\n(c={c:.0f})", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 오른쪽: log 스케일 RT vs CFT 정확도
    ax = axes[1]
    l_log = np.logspace(-2, 2, 200)
    S_cft = ee_vacuum(l_log, c, epsilon)
    S_rt = np.array([holographic_ee(ads, li, epsilon) for li in l_log])

    ax.loglog(l_log, S_cft, color=PALETTE["uv"], lw=3, label="CFT (Calabrese-Cardy)", zorder=3)
    ax.loglog(l_log, S_rt, color=PALETTE["ir"], lw=1.5, ls="--",
              label="홀로그래픽 RT", zorder=2)

    ax.set_xlabel("구간 길이 l (log 척도)", fontsize=11)
    ax.set_ylabel("S (log 척도)", fontsize=11)
    ax.set_title(r"RT 공식 정밀도: $S_{RT}/S_{CFT} = 1$ (기계 정밀도)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 오차 인셋
    ins = ax.inset_axes([0.55, 0.05, 0.4, 0.35])
    rel_err = np.abs(S_cft - S_rt) / (np.abs(S_cft) + 1e-15)
    ins.semilogy(l_log, rel_err + 1e-17, color=PALETTE["accent"], lw=1.5)
    ins.set_title("상대 오차", fontsize=7)
    ins.set_xlabel("l", fontsize=6)
    ins.tick_params(labelsize=6)
    ins.set_ylim(1e-17, 1e-12)
    ins.grid(alpha=0.3)

    fig.tight_layout()
    path = save_path or save(fig, "02_ee_comparison.png")
    return path


# ── 3. 홀로그래픽 위상 전이 ───────────────────────────────────

def plot_phase_transition(save_path: Optional[str] = None) -> str:
    ads = AdS3(L=1.0, G_N=0.05)
    epsilon = 1e-3

    l_values = [0.5, 1.0, 2.0, 4.0]
    gaps = np.linspace(0.01, 8.0, 300)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 왼쪽: 여러 l 값에 대한 I(A:B) vs gap
    ax = axes[0]
    colors = [PALETTE["uv"], PALETTE["accent"], PALETTE["mid"], PALETTE["ir"]]
    for l, color in zip(l_values, colors):
        I_vals = []
        phases = []
        for d in gaps:
            A = (-l - d/2, -d/2)
            B = (d/2, l + d/2)
            I, phase = mutual_information_holographic(ads, A, B, epsilon)
            I_vals.append(I)
            phases.append(phase)
        I_vals = np.array(I_vals)
        ax.plot(gaps, I_vals, color=color, lw=2.5, label=f"l={l}")
        # 위상 전이점 표시
        trans = np.where(np.diff((np.array(phases) == "disconnected").astype(int)) > 0)[0]
        if len(trans):
            ax.axvline(gaps[trans[0]], color=color, ls=":", alpha=0.5, lw=1)

    ax.set_xlabel("구간 간격 d", fontsize=11)
    ax.set_ylabel("상호 정보량 I(A:B)", fontsize=11)
    ax.set_title("홀로그래픽 1차 위상 전이\n(Connected → Disconnected RT 면)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 50],
                     0, 0.001, alpha=0.1, color=PALETTE["uv"])

    # 오른쪽: 위상도 (l, d 공간)
    ax = axes[1]
    l_scan = np.linspace(0.1, 5.0, 60)
    d_scan = np.linspace(0.01, 5.0, 60)
    L_grid, D_grid = np.meshgrid(l_scan, d_scan)
    Phase_grid = np.zeros_like(L_grid)

    for i in range(len(d_scan)):
        for j in range(len(l_scan)):
            l = L_grid[i, j]
            d = D_grid[i, j]
            A = (-l - d/2, -d/2)
            B = (d/2, l + d/2)
            _, phase = mutual_information_holographic(ads, A, B, epsilon)
            Phase_grid[i, j] = 1.0 if phase == "connected" else 0.0

    im = ax.contourf(L_grid, D_grid, Phase_grid, levels=[0, 0.5, 1],
                     colors=[PALETTE["light"], "#AED6F1"], alpha=0.8)
    ax.contour(L_grid, D_grid, Phase_grid, levels=[0.5],
               colors=[PALETTE["dark"]], linewidths=2)

    # 이론 경계선 d* = (√5-2) l
    l_theory = np.linspace(0.1, 5.0, 100)
    d_theory = (np.sqrt(5) - 2) * l_theory
    ax.plot(l_theory, d_theory * 2, color=PALETTE["uv"], lw=2, ls="--",
            label=f"이론: d* = {np.sqrt(5)-2:.3f}l")

    patches = [mpatches.Patch(color="#AED6F1", label="I(A:B) > 0\n(Connected)"),
               mpatches.Patch(color=PALETTE["light"], label="I(A:B) = 0\n(Disconnected)")]
    ax.legend(handles=patches, fontsize=8, loc="upper left")
    ax.set_xlabel("구간 길이 l", fontsize=11)
    ax.set_ylabel("간격 d", fontsize=11)
    ax.set_title("홀로그래픽 위상도 (l-d 공간)", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = save_path or save(fig, "03_phase_transition.png")
    return path


# ── 4. Page 곡선 ──────────────────────────────────────────────

def plot_page_curve(save_path: Optional[str] = None) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 여러 온도의 BTZ
    for ax_idx, (M_val, label) in enumerate([(0.5, "가벼운 BH (M=0.5)"),
                                              (2.0, "무거운 BH (M=2.0)")]):
        ax = axes[ax_idx]
        btz = BTZBlackHole(M=M_val, l=1.0, G_N=0.05)
        island = EntanglementIsland(btz)

        t_max = btz.page_time() * 2.5
        t_vals = np.linspace(0.01, t_max, 400)
        result = island.page_curve(t_vals, epsilon=1e-3)

        ax.plot(t_vals, result["S_no_island"], color=PALETTE["uv"], lw=2.5,
                label="S (Island 없음, Hawking)", zorder=3)
        ax.axhline(result["S_island"][0], color=PALETTE["ir"], lw=2.5,
                   ls="--", label="S (Island 포함)")
        ax.plot(t_vals, result["S_page"], color=PALETTE["dark"], lw=3,
                label="Page 곡선 (min)", zorder=4)

        # Page 시간 표시
        t_p = result["t_page"]
        if not np.isinf(t_p):
            ax.axvline(t_p, color=PALETTE["accent"], ls=":", lw=2,
                       label=f"Page 시간 t_P={t_p:.1f}")
            ax.scatter([t_p], [result["S_BH"]], color=PALETTE["accent"], s=100, zorder=5)

        # Scrambling 시간
        t_sc = result["t_scrambling"]
        if not np.isinf(t_sc) and t_sc < t_max:
            ax.axvline(t_sc, color=PALETTE["mid"], ls="--", lw=1.5, alpha=0.7,
                       label=f"Scrambling t_*={t_sc:.2f}")

        ax.set_xlabel("시간 t", fontsize=11)
        ax.set_ylabel("S(복사 영역 R)", fontsize=11)
        ax.set_title(f"Page 곡선 — Island 공식\n{label}\n"
                     f"T_H={btz.hawking_temperature:.3f}, S_BH={result['S_BH']:.1f}",
                     fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
        ax.set_ylim(0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # 주석
        ax.text(0.02, 0.6, "Hawking\n복사", transform=ax.transAxes,
                color=PALETTE["uv"], fontsize=8, alpha=0.8)
        ax.text(0.65, 0.2, "Island\n위상", transform=ax.transAxes,
                color=PALETTE["ir"], fontsize=8, alpha=0.8)

    fig.suptitle("블랙홀 정보 역설: Island 공식 & Page 곡선", fontsize=14, y=1.02)
    fig.tight_layout()
    path = save_path or save(fig, "04_page_curve.png")
    return path


# ── 5. MERA 네트워크 다이어그램 ───────────────────────────────

def plot_mera_network(n_sites: int = 8, save_path: Optional[str] = None) -> str:
    n_layers = int(np.log2(n_sites))
    fig, ax = plt.subplots(figsize=(11, 7))

    node_positions = {}
    for layer in range(n_layers + 1):
        n = n_sites // (2 ** layer)
        x_positions = np.linspace(-(n - 1) / 2, (n - 1) / 2, n)
        y = layer * 1.5
        for i, x in enumerate(x_positions):
            node_positions[(layer, i)] = (x, y)

    # 등척사상(isometry) — 위로 향하는 연결
    for layer in range(n_layers):
        n = n_sites // (2 ** layer)
        for i in range(n // 2):
            parent = (layer + 1, i)
            child1 = (layer, 2 * i)
            child2 = (layer, 2 * i + 1)
            x_p, y_p = node_positions[parent]
            x1, y1 = node_positions[child1]
            x2, y2 = node_positions[child2]
            # 이소메트리 삼각형
            tri = plt.Polygon([[x1, y1], [x2, y2], [x_p, y_p - 0.3]],
                               fill=True, facecolor="#AED6F1", edgecolor="#2980B9",
                               lw=1.5, alpha=0.7, zorder=2)
            ax.add_patch(tri)

    # 디스인탱글러 — 같은 층에서 인접 노드 연결
    for layer in range(n_layers):
        n = n_sites // (2 ** layer)
        for i in range(0, n - 1, 2):
            x1, y1 = node_positions[(layer, i)]
            x2, y2 = node_positions[(layer, i + 1)]
            x_m, y_m = (x1 + x2) / 2, y1 + 0.25
            sq = plt.Polygon([[x1, y1], [x_m, y_m + 0.15],
                               [x2, y2], [x_m, y_m - 0.15]],
                              fill=True, facecolor="#FAD7A0", edgecolor="#E67E22",
                              lw=1.5, alpha=0.8, zorder=3)
            ax.add_patch(sq)

    # 노드
    for (layer, i), (x, y) in node_positions.items():
        n = n_sites // (2 ** layer)
        if layer == 0:
            color = PALETTE["uv"]
        elif layer == n_layers:
            color = PALETTE["ir"]
        else:
            color = PALETTE["mid"]
        circle = plt.Circle((x, y), 0.18, color=color, zorder=4, ec="white", lw=1.5)
        ax.add_patch(circle)

    # AdS z 좌표 레이블
    for layer in range(n_layers + 1):
        z_ads = 1.0 * 2 ** layer / n_sites
        ax.text(n_sites / 2 + 0.4, layer * 1.5, f"z={z_ads:.3f}", fontsize=8,
                color="gray", va="center")

    ax.text(n_sites / 2 + 0.4, (n_layers + 0.5) * 1.5, "AdS z↑", fontsize=9,
            color=PALETTE["dark"], va="center", fontweight="bold")

    # 범례
    legend_elements = [
        mpatches.Patch(color=PALETTE["uv"], label="UV 경계 사이트 (CFT 자유도)"),
        mpatches.Patch(color=PALETTE["mid"], label="벌크 사이트 (중간 에너지)"),
        mpatches.Patch(color=PALETTE["ir"], label="IR 고정점 (최저 에너지)"),
        mpatches.Patch(color="#AED6F1", label="등척사상 (isometry) = 벌크 전파자"),
        mpatches.Patch(color="#FAD7A0", label="디스인탱글러 = RT 측지선 절단"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.9)

    ax.set_xlim(-n_sites / 2 - 0.5, n_sites / 2 + 1.5)
    ax.set_ylim(-0.5, n_layers * 1.5 + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"MERA 텐서 네트워크 ({n_sites} 사이트) ↔ AdS 기하학\n"
                 "Swingle (2012): MERA의 인과 원뿔 경계 = Ryu-Takayanagi 측지선",
                 fontsize=12)

    path = save_path or save(fig, "05_mera_network.png")
    return path


# ── 6. 홀로그래픽 RG 흐름 ─────────────────────────────────────

def plot_rg_flow(save_path: Optional[str] = None) -> str:
    from .tensor_network import MERA
    from ml.holographic_rg import HolographicRGNetwork

    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, fig, wspace=0.35)

    # 왼쪽: c-함수
    ax1 = fig.add_subplot(gs[0])
    ads = AdS3(L=1.0, G_N=0.05)
    c_uv = ads.central_charge
    rng = np.random.default_rng(42)

    net = HolographicRGNetwork(dim=6, n_layers=30, z_uv=0.01, z_ir=20.0, seed=7)
    phi_uv = rng.normal(0, 0.5, 6)
    c_vals = net.c_function(phi_uv, c_uv)

    log_z = np.linspace(np.log(net.z_uv), np.log(net.z_ir), 31)
    z_vals = np.exp(log_z)
    mu_vals = 1.0 / z_vals  # 에너지 척도 μ ~ 1/z

    ax1.plot(np.log10(mu_vals), c_vals, color=PALETTE["uv"], lw=2.5)
    ax1.axhline(c_uv, color="gray", ls="--", alpha=0.5, label=f"$c_{{UV}}={c_uv:.0f}$")
    ax1.axhline(0, color="gray", ls=":", alpha=0.5, label="$c_{IR}=0$")
    ax1.set_xlabel(r"$\log_{10}(\mu)$  [에너지 척도]", fontsize=10)
    ax1.set_ylabel("c-함수", fontsize=10)
    ax1.set_title("Zamolodchikov c-정리\n(RG 흐름 따라 c 단조 감소)", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.annotate("UV\n(고에너지)", xy=(np.log10(mu_vals[0]), c_vals[0]),
                 xytext=(np.log10(mu_vals[0]) - 0.3, c_vals[0] * 0.7),
                 fontsize=8, color=PALETTE["uv"],
                 arrowprops=dict(arrowstyle="->", color=PALETTE["uv"]))
    ax1.annotate("IR\n(저에너지)", xy=(np.log10(mu_vals[-1]), c_vals[-1]),
                 xytext=(np.log10(mu_vals[-1]) - 0.5, c_vals[-1] + c_uv * 0.2),
                 fontsize=8, color=PALETTE["ir"],
                 arrowprops=dict(arrowstyle="->", color=PALETTE["ir"]))

    # 중간: 층별 얽힘 엔트로피
    ax2 = fig.add_subplot(gs[1])
    net2 = HolographicRGNetwork(dim=8, n_layers=20, z_uv=0.01, z_ir=10.0, seed=3)
    phi2 = rng.normal(0, 1.0, 8)
    entropies = net2.entanglement_entropy_estimate(phi2)

    log_z2 = np.linspace(np.log(net2.z_uv), np.log(net2.z_ir), 21)
    z2 = np.exp(log_z2)

    ax2.plot(z2, entropies, "o-", color=PALETTE["mid"], lw=2, ms=5, label="신경망 S(z)")
    ax2.set_xlabel("AdS 지름 방향 z", fontsize=10)
    ax2.set_ylabel("층별 얽힘 엔트로피", fontsize=10)
    ax2.set_title("신경망 층 → 얽힘 엔트로피\n(층 깊이 ↔ AdS z 방향)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # 오른쪽: MERA 층별 EE vs RT
    ax3 = fig.add_subplot(gs[2])
    mera = MERA(n_sites=64, L_ads=1.0, G_N=0.05)
    l_sites = np.arange(1, 33)
    l_phys = l_sites * 2.0 / 64
    epsilon_phys = 2.0 / 64

    S_mera = [mera.entanglement_entropy_mera(int(l), 1) for l in l_sites]
    S_rt = [ads.ryu_takayanagi_entropy(-lp/2, lp/2, epsilon_phys) for lp in l_phys]

    ax3.plot(np.log2(l_sites), S_mera, "s-", color=PALETTE["accent"], lw=2, ms=5, label="MERA")
    ax3.plot(np.log2(l_sites), S_rt, "o--", color=PALETTE["ir"], lw=2, ms=5, label="RT 공식")
    ax3.set_xlabel(r"$\log_2(l)$  [구간 크기]", fontsize=10)
    ax3.set_ylabel("얽힘 엔트로피 S", fontsize=10)
    ax3.set_title(r"MERA $\leftrightarrow$ RT: $S \propto \log(l)$", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    path = save_path or save(fig, "06_rg_flow.png")
    return path


# ── 7. 벌크 장 프로파일 ───────────────────────────────────────

def plot_bulk_profile(save_path: Optional[str] = None) -> str:
    from .geometry import AdS3

    ads = AdS3(L=1.0, G_N=0.05)
    x_grid = np.linspace(-4, 4, 120)
    z_grid = np.linspace(0.05, 3.5, 80)
    delta = 2.0

    # 가우시안 소스
    source = np.exp(-x_grid**2 / (2 * 0.5**2))
    dx = x_grid[1] - x_grid[0]

    # 벌크 장: φ(z, x) = ∫ K(z,x;x') J(x') dx'
    phi = np.zeros((len(z_grid), len(x_grid)))
    for iz, z in enumerate(z_grid):
        K = ads.bulk_to_boundary_propagator(z, x_grid[:, None], x_grid[None, :], delta)
        phi[iz] = K @ source * dx

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 왼쪽: 2D 프로파일
    ax = axes[0]
    Z, X = np.meshgrid(z_grid, x_grid, indexing="ij")
    im = ax.contourf(X, Z, phi, levels=40, cmap="RdYlBu_r")
    plt.colorbar(im, ax=ax, label="φ(z, x)")
    ax.set_xlabel("경계 좌표 x", fontsize=11)
    ax.set_ylabel("벌크 방향 z", fontsize=11)
    ax.set_title(f"HKLL 벌크 재건: φ(z, x)\n"
                 r"$\phi(z,x) = \int dx'\, K_\Delta(z,x;x')\, J(x')$" + f"\n(Δ={delta})",
                 fontsize=11)
    ax.invert_yaxis()
    ax.text(0.02, 0.02, "UV 경계 (z→0)", transform=ax.transAxes, fontsize=8, color="white")
    ax.text(0.02, 0.92, "IR 깊은 벌크", transform=ax.transAxes, fontsize=8, color="white")

    # 오른쪽: 여러 z에서의 단면
    ax = axes[1]
    colors_z = [PALETTE["uv"], PALETTE["accent"], PALETTE["mid"], PALETTE["ir"]]
    z_samples = [0.1, 0.5, 1.0, 2.0]
    for z_s, color in zip(z_samples, colors_z):
        iz = np.argmin(np.abs(z_grid - z_s))
        ax.plot(x_grid, phi[iz], color=color, lw=2, label=f"z={z_s}")

    ax.plot(x_grid, source * phi[:, len(x_grid)//2].max() / source.max(),
            "k--", lw=1.5, alpha=0.5, label="경계 소스 J(x)")
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("φ(z, x)", fontsize=11)
    ax.set_title("깊이별 벌크 장 프로파일\n(UV→IR로 갈수록 퍼짐)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = save_path or save(fig, "07_bulk_profile.png")
    return path


# ── 8. BTZ 블랙홀 열역학 ─────────────────────────────────────

def plot_btz_thermodynamics(save_path: Optional[str] = None) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (1, 1): T_H vs M (Hawking 온도)
    ax = axes[0, 0]
    M_vals = np.linspace(0.01, 5.0, 200)
    for l_val, color in [(0.5, PALETTE["uv"]), (1.0, PALETTE["ir"]), (2.0, PALETTE["mid"])]:
        T_H = [BTZBlackHole(M=m, l=l_val, G_N=0.05).hawking_temperature for m in M_vals]
        ax.plot(M_vals, T_H, color=color, lw=2, label=f"l={l_val}")
    ax.set_xlabel("질량 M", fontsize=10)
    ax.set_ylabel("Hawking 온도 T_H", fontsize=10)
    ax.set_title(r"$T_H = \sqrt{M}/(2\pi l)$", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (1, 2): S_BH vs M
    ax = axes[0, 1]
    for l_val, color in [(0.5, PALETTE["uv"]), (1.0, PALETTE["ir"]), (2.0, PALETTE["mid"])]:
        S_BH = [BTZBlackHole(M=m, l=l_val, G_N=0.05).bekenstein_hawking_entropy for m in M_vals]
        ax.plot(M_vals, S_BH, color=color, lw=2, label=f"l={l_val}")
    ax.set_xlabel("질량 M", fontsize=10)
    ax.set_ylabel("BH 엔트로피 S_BH", fontsize=10)
    ax.set_title(r"$S_{BH} = \pi c r_+ / 3$  (Brown-Henneaux)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (2, 1): 회전 BTZ (J ≠ 0)
    ax = axes[1, 0]
    M0 = 2.0
    J_vals = np.linspace(0, M0 * 0.99, 100)
    btz_list = [BTZBlackHole(M=M0, l=1.0, G_N=0.05, J=j) for j in J_vals]
    T_rot = [b.hawking_temperature for b in btz_list]
    S_rot = [b.bekenstein_hawking_entropy for b in btz_list]
    ax.plot(J_vals, T_rot, color=PALETTE["uv"], lw=2.5, label="T_H")
    ax2_twin = ax.twinx()
    ax2_twin.plot(J_vals, S_rot, color=PALETTE["ir"], lw=2.5, ls="--", label="S_BH")
    ax.set_xlabel("각운동량 J", fontsize=10)
    ax.set_ylabel("Hawking 온도 T_H", fontsize=10, color=PALETTE["uv"])
    ax2_twin.set_ylabel("BH 엔트로피 S_BH", fontsize=10, color=PALETTE["ir"])
    ax.set_title(f"회전 BTZ 블랙홀 (M={M0})\n|J| → Ml 극한: 극한 회전 (T_H → 0)", fontsize=10)
    ax.tick_params(axis="y", labelcolor=PALETTE["uv"])
    ax2_twin.tick_params(axis="y", labelcolor=PALETTE["ir"])
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)

    # (2, 2): Lyapunov 지수와 MSS bound
    ax = axes[1, 1]
    M_range = np.linspace(0.1, 5.0, 100)
    btz_arr = [BTZBlackHole(M=m, l=1.0, G_N=0.05) for m in M_range]
    lyapunov = [b.lyapunov_exponent() for b in btz_arr]
    mss_bound = [2 * np.pi * b.hawking_temperature for b in btz_arr]

    ax.plot(M_range, lyapunov, color=PALETTE["uv"], lw=2.5, label=r"$\lambda_L = 2\pi T_H$")
    ax.plot(M_range, mss_bound, color=PALETTE["ir"], lw=1.5, ls="--",
            label="MSS bound", alpha=0.7)
    ax.fill_between(M_range, lyapunov, mss_bound, alpha=0.1, color=PALETTE["accent"])
    ax.set_xlabel("질량 M", fontsize=10)
    ax.set_ylabel(r"Lyapunov 지수 $\lambda_L$", fontsize=10)
    ax.set_title(r"양자 혼돈: $\lambda_L \leq 2\pi T_H$ (MSS bound)" + "\nBTZ는 bound 포화 → 최대 혼돈 시스템!", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("BTZ 블랙홀 홀로그래픽 열역학", fontsize=14, y=1.01)
    fig.tight_layout()
    path = save_path or save(fig, "08_btz_thermodynamics.png")
    return path


def plot_all() -> List[str]:
    """모든 그래프를 생성하고 경로 목록 반환."""
    print("  그래프 생성 중...")
    paths = []
    tasks = [
        ("AdS₃ 측지선", plot_geodesics),
        ("RT vs CFT 비교", plot_ee_comparison),
        ("홀로그래픽 위상 전이", plot_phase_transition),
        ("Page 곡선", plot_page_curve),
        ("MERA 네트워크", plot_mera_network),
        ("홀로그래픽 RG 흐름", plot_rg_flow),
        ("벌크 장 프로파일", plot_bulk_profile),
        ("BTZ 열역학", plot_btz_thermodynamics),
    ]
    for name, fn in tasks:
        try:
            path = fn()
            paths.append(path)
            print(f"  ✓ {name}: {os.path.basename(path)}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    return paths


