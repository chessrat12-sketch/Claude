"""
PyTorch 기반 홀로그래픽 ML 학습 루프

학습 목표:
  1. BulkBoundaryNet: CFT 경계 데이터 → 벌크 장 재건
  2. BetaFunctionNet: β(g) 함수 역공학 (RG 흐름 학습)
  3. HolographicEENet: 경계 상태 → 얽힘 엔트로피 예측
  4. RTGeometryNet: 벌크 측지선 길이 학습

모든 네트워크는 AdS/CFT 물리를 손실 함수에 인코딩.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from ads_cft_ml.geometry import AdS3
from ads_cft_ml.entanglement import ee_vacuum, holographic_ee


# ── 유틸리티 ─────────────────────────────────────────────────

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ═══════════════════════════════════════════════════════════════
# 1. CFT 데이터셋
# ═══════════════════════════════════════════════════════════════

class CFTBoundaryDataset(Dataset):
    """
    CFT 경계 데이터: (소스 J(x), 1점 함수 <O(x)>) 쌍.

    소스 J(x): 가우시안 혼합 (다양한 소스 형태)
    <O(x)>: CFT Green 함수로 계산
              <O(x)> ∝ ∫ |x-x'|^{-2Δ} J(x') dx'
    """
    def __init__(
        self,
        n_samples: int = 2000,
        n_sites: int = 64,
        delta: float = 2.0,
        x_range: float = 5.0,
        seed: int = 0,
    ):
        self.n_samples = n_samples
        self.n_sites = n_sites
        self.delta = delta
        self.x_grid = np.linspace(-x_range, x_range, n_sites)
        self.dx = self.x_grid[1] - self.x_grid[0]

        rng = np.random.default_rng(seed)
        self.sources = []
        self.one_point_fns = []

        # Green 함수 행렬 (정규화된)
        G = self._green_function_matrix()

        for _ in range(n_samples):
            # 다양한 소스: 가우시안 혼합
            n_peaks = rng.integers(1, 4)
            J = np.zeros(n_sites)
            for _ in range(n_peaks):
                center = rng.uniform(-x_range * 0.7, x_range * 0.7)
                width = rng.uniform(0.3, 1.5)
                amp = rng.uniform(-1.0, 1.0)
                J += amp * np.exp(-0.5 * ((self.x_grid - center) / width) ** 2)

            O = G @ J * self.dx
            self.sources.append(J.astype(np.float32))
            self.one_point_fns.append(O.astype(np.float32))

        self.sources = np.array(self.sources)
        self.one_point_fns = np.array(self.one_point_fns)

    def _green_function_matrix(self) -> np.ndarray:
        """CFT 2점 함수: G_{ij} = |x_i - x_j|^{-2Δ}"""
        G = np.zeros((self.n_sites, self.n_sites))
        for i in range(self.n_sites):
            for j in range(self.n_sites):
                d = abs(self.x_grid[i] - self.x_grid[j])
                G[i, j] = d ** (-2 * self.delta) if d > 0.1 else 0.0
        # 정규화
        G /= G.max() + 1e-10
        return G

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.sources[idx]),
            torch.tensor(self.one_point_fns[idx]),
        )


class EntanglementDataset(Dataset):
    """
    (경계 상태 벡터, 얽힘 엔트로피) 쌍 데이터셋.
    RT 공식으로 정답 생성.
    """
    def __init__(
        self,
        n_samples: int = 3000,
        n_intervals: int = 20,
        l_max: float = 8.0,
        epsilon: float = 1e-3,
        L_ads: float = 1.0,
        G_N: float = 0.05,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        ads = AdS3(L=L_ads, G_N=G_N)
        self.c = ads.central_charge

        # 입력: (l, log(l/ε)) 특성 벡터
        # 출력: RT 얽힘 엔트로피 S(l)
        l_values = rng.uniform(epsilon * 2, l_max, (n_samples, n_intervals))
        S_values = np.zeros_like(l_values)

        for i in range(n_samples):
            for j in range(n_intervals):
                S_values[i, j] = holographic_ee(ads, l_values[i, j], epsilon)

        # 특성: [l, l², log(l), 1]
        log_l = np.log(l_values / epsilon)
        features = np.stack([
            l_values,
            l_values ** 2,
            log_l,
            np.ones_like(l_values),
        ], axis=-1).reshape(n_samples, n_intervals * 4)

        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(S_values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ═══════════════════════════════════════════════════════════════
# 2. 신경망 모델
# ═══════════════════════════════════════════════════════════════

class BulkBoundaryNet(nn.Module):
    """
    경계 소스 J(x) → 벌크 장 φ(z_layers, x) 매핑 학습.

    각 출력 채널 = 다른 z 깊이에서의 벌크 장.
    구조: 1D Conv (비국소 상호작용) + RG 흐름 레이어
    """
    def __init__(self, n_sites: int = 64, n_z_layers: int = 8, hidden: int = 128):
        super().__init__()
        self.n_sites = n_sites
        self.n_z_layers = n_z_layers

        # 비국소 경계 처리 (상관 함수 구조)
        self.boundary_processor = nn.Sequential(
            nn.Linear(n_sites, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

        # 각 z-층으로의 투영 (RG 흐름)
        self.z_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.Tanh(),
                nn.Linear(hidden // 2, n_sites),
            )
            for _ in range(n_z_layers)
        ])

        # 경계 재건 (1점 함수 예측)
        self.boundary_readout = nn.Linear(n_sites, n_sites)

    def forward(self, J: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        J: (batch, n_sites)
        반환: {
          'one_point_fn': (batch, n_sites),
          'bulk_profile': (batch, n_z_layers, n_sites),
        }
        """
        h = self.boundary_processor(J)
        bulk = torch.stack([proj(h) for proj in self.z_projectors], dim=1)
        one_pt = self.boundary_readout(bulk[:, -1])
        return {"one_point_fn": one_pt, "bulk_profile": bulk}


class BetaFunctionNet(nn.Module):
    """
    β 함수 신경망: β(g) = dg/d(log μ)

    입력: 결합 상수 벡터 g
    출력: β(g)

    물리적 제약:
    - β(g*) = 0  에서 RG 고정점
    - c-정리: β 방향 따라 c 감소
    """
    def __init__(self, n_couplings: int = 4, hidden: int = 64):
        super().__init__()
        self.n_couplings = n_couplings
        self.net = nn.Sequential(
            nn.Linear(n_couplings, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, n_couplings),
        )
        # 초기화: β(0) ≈ 0 (근사 고정점)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return self.net(g)

    def rg_flow(self, g0: torch.Tensor, n_steps: int = 50, dlogmu: float = 0.1) -> torch.Tensor:
        """RG 흐름 적분: UV → IR"""
        g = g0.clone()
        trajectory = [g.clone()]
        for _ in range(n_steps):
            with torch.no_grad():
                beta = self.forward(g)
                g = g - dlogmu * beta  # μ 감소 = z 증가
            trajectory.append(g.clone())
        return torch.stack(trajectory)


class HolographicEENet(nn.Module):
    """
    (구간 길이 l) → 얽힘 엔트로피 S(l) 학습.

    물리적 사전 지식을 활용한 구조:
    - 입력 특성: [l, log(l), l², 1]
    - RT 공식 S = (c/3) log(l/ε) 를 근사하도록 유도
    - 잔차 연결: S = S_RT(l) + δS(l)  [보정항 학습]
    """
    def __init__(self, n_intervals: int = 20, hidden: int = 128,
                 c_prior: float = 30.0, epsilon: float = 1e-3):
        super().__init__()
        self.c_prior = c_prior
        self.epsilon = epsilon
        self.n_intervals = n_intervals
        in_dim = n_intervals * 4

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, n_intervals),
        )
        # 학습 가능한 c (중심 전하)
        self.log_c = nn.Parameter(torch.log(torch.tensor(c_prior)))

    @property
    def c(self) -> torch.Tensor:
        return torch.exp(self.log_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_intervals * 4) — 특성 벡터
        반환: (batch, n_intervals) — 얽힘 엔트로피
        """
        # 로그 특성 추출 (n번째 특성: log(l/ε))
        log_l = x[:, 2::4]  # log(l/ε) 특성들

        # RT 공식 (선형 부분)
        S_rt = (self.c / 3.0) * log_l

        # 비선형 보정
        delta_S = self.trunk(x)

        return S_rt + 0.1 * delta_S  # 보정은 작게 시작


# ═══════════════════════════════════════════════════════════════
# 3. 손실 함수
# ═══════════════════════════════════════════════════════════════

class HolographicLoss(nn.Module):
    """
    물리적 제약을 반영한 홀로그래픽 손실 함수.

    L = L_reconstruct + λ_EE * L_EE + λ_mono * L_monotone + λ_c * L_ctheorem
    """
    def __init__(
        self,
        lambda_ee: float = 0.5,
        lambda_mono: float = 0.1,
        lambda_c: float = 0.05,
    ):
        super().__init__()
        self.lambda_ee = lambda_ee
        self.lambda_mono = lambda_mono
        self.lambda_c = lambda_c

    def reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(pred, target)

    def monotonicity_loss(self, S_pred: torch.Tensor, l_features: torch.Tensor) -> torch.Tensor:
        """
        물리적 제약: S(l) 단조 증가 (l이 클수록 EE가 더 큼).
        위반 시 패널티.
        """
        dS = torch.diff(S_pred, dim=-1)
        violations = torch.clamp(-dS, min=0)  # 감소하는 경우 패널티
        return violations.mean()

    def c_theorem_loss(self, bulk_profile: torch.Tensor) -> torch.Tensor:
        """
        c-정리 제약: 층 깊이 따라 정보가 감소.
        ||φ(z_{k+1})|| ≤ ||φ(z_k)||
        """
        norms = bulk_profile.norm(dim=-1)  # (batch, n_z)
        diffs = torch.diff(norms, dim=-1)   # (batch, n_z - 1)
        violations = torch.clamp(diffs, min=0)  # 증가하는 경우 패널티
        return violations.mean()

    def forward(
        self,
        pred_one_pt: torch.Tensor,
        target_one_pt: torch.Tensor,
        bulk_profile: Optional[torch.Tensor] = None,
        S_pred: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        losses["reconstruct"] = self.reconstruction_loss(pred_one_pt, target_one_pt)
        total = losses["reconstruct"]

        if bulk_profile is not None:
            losses["c_theorem"] = self.c_theorem_loss(bulk_profile)
            total = total + self.lambda_c * losses["c_theorem"]

        if S_pred is not None:
            losses["monotone"] = self.monotonicity_loss(S_pred, None)
            total = total + self.lambda_mono * losses["monotone"]

        losses["total"] = total
        return losses


# ═══════════════════════════════════════════════════════════════
# 4. 학습 루프
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    lr: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 30
    weight_decay: float = 1e-4
    scheduler_patience: int = 5
    device: str = "cpu"


def train_bulk_boundary(
    config: TrainConfig = TrainConfig(),
    n_samples: int = 1000,
    n_sites: int = 32,
    verbose: bool = True,
) -> Tuple[BulkBoundaryNet, List[float], List[float]]:
    """
    BulkBoundaryNet 학습.
    CFT 소스 J(x) → 1점 함수 <O(x)> 매핑.
    """
    set_seed(42)
    device = torch.device(config.device)

    dataset = CFTBoundaryDataset(n_samples=n_samples, n_sites=n_sites)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    model = BulkBoundaryNet(n_sites=n_sites, n_z_layers=6, hidden=64).to(device)
    loss_fn = HolographicLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr,
                            weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=config.scheduler_patience, factor=0.5,
    )

    train_losses, val_losses = [], []

    for epoch in range(config.n_epochs):
        # 학습
        model.train()
        epoch_loss = 0.0
        for J, O_target in train_loader:
            J, O_target = J.to(device), O_target.to(device)
            optimizer.zero_grad()
            out = model(J)
            loss_dict = loss_fn(
                out["one_point_fn"], O_target,
                bulk_profile=out["bulk_profile"],
            )
            loss_dict["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss_dict["total"].item()

        # 검증
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for J, O_target in val_loader:
                J, O_target = J.to(device), O_target.to(device)
                out = model(J)
                loss_dict = loss_fn(out["one_point_fn"], O_target,
                                   bulk_profile=out["bulk_profile"])
                val_loss += loss_dict["total"].item()

        avg_train = epoch_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if verbose and (epoch % 5 == 0 or epoch == config.n_epochs - 1):
            print(f"    에폭 {epoch+1:3d}/{config.n_epochs}  "
                  f"학습={avg_train:.4f}  검증={avg_val:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

    return model, train_losses, val_losses


def train_ee_network(
    config: TrainConfig = TrainConfig(),
    n_samples: int = 2000,
    n_intervals: int = 15,
    verbose: bool = True,
) -> Tuple[HolographicEENet, List[float], List[float], float]:
    """
    HolographicEENet 학습.
    학습된 중심 전하 c와 RT 공식 비교.
    """
    set_seed(0)
    device = torch.device(config.device)

    ads = AdS3(L=1.0, G_N=0.05)
    true_c = ads.central_charge

    dataset = EntanglementDataset(
        n_samples=n_samples, n_intervals=n_intervals,
        L_ads=1.0, G_N=0.05,
    )
    n_train = int(0.8 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, len(dataset) - n_train],
        generator=torch.Generator().manual_seed(1),
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    model = HolographicEENet(n_intervals=n_intervals, hidden=64, c_prior=true_c * 0.5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr,
                            weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_epochs)

    train_losses, val_losses = [], []

    for epoch in range(config.n_epochs):
        model.train()
        epoch_loss = 0.0
        for X, y_true in train_loader:
            X, y_true = X.to(device), y_true.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = nn.functional.mse_loss(y_pred, y_true)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y_true in val_loader:
                X, y_true = X.to(device), y_true.to(device)
                y_pred = model(X)
                val_loss += nn.functional.mse_loss(y_pred, y_true).item()

        avg_train = epoch_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step()

        if verbose and (epoch % 5 == 0 or epoch == config.n_epochs - 1):
            learned_c = model.c.item()
            print(f"    에폭 {epoch+1:3d}/{config.n_epochs}  "
                  f"학습={avg_train:.5f}  검증={avg_val:.5f}  "
                  f"학습된 c={learned_c:.2f} (정답={true_c:.1f})")

    learned_c = model.c.item()
    return model, train_losses, val_losses, learned_c


def train_beta_function(
    config: TrainConfig = TrainConfig(),
    n_couplings: int = 3,
    verbose: bool = True,
) -> Tuple[BetaFunctionNet, List[float]]:
    """
    β 함수 학습: 알려진 RG 흐름 궤적으로부터 β(g) 역공학.

    목표 β 함수: 이차원 Ising 모형 근방 (단순화된)
    β(g₁, g₂, g₃) = (-ε g₁ + g₁³, (2-ε)g₂ - g₁²g₂, g₃ - g₂²)
    """
    set_seed(3)
    device = torch.device(config.device)

    def true_beta(g: np.ndarray) -> np.ndarray:
        """목표 β 함수 (Ising 임계점 근방 단순화)"""
        eps = 0.1
        b = np.zeros(n_couplings)
        b[0] = -eps * g[0] + g[0]**3
        if n_couplings > 1:
            b[1] = (2 - eps) * g[1] - g[0]**2 * g[1]
        if n_couplings > 2:
            b[2] = g[2] - g[1]**2
        return b

    # RG 궤적 생성
    rng = np.random.default_rng(42)
    n_traj = 500
    n_steps = 20
    dlogmu = 0.05

    g_data, beta_data = [], []
    for _ in range(n_traj):
        g = rng.uniform(-1.0, 1.0, n_couplings)
        for _ in range(n_steps):
            beta = true_beta(g)
            g_data.append(g.astype(np.float32))
            beta_data.append(beta.astype(np.float32))
            g = g - dlogmu * beta  # 다음 스텝

    X = torch.tensor(np.array(g_data), dtype=torch.float32).to(device)
    y = torch.tensor(np.array(beta_data), dtype=torch.float32).to(device)

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = BetaFunctionNet(n_couplings=n_couplings, hidden=48).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    losses = []
    for epoch in range(config.n_epochs):
        model.train()
        epoch_loss = 0.0
        for g_batch, beta_batch in loader:
            optimizer.zero_grad()
            beta_pred = model(g_batch)
            loss = nn.functional.mse_loss(beta_pred, beta_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        losses.append(avg)

        if verbose and (epoch % 5 == 0 or epoch == config.n_epochs - 1):
            # 고정점 검사
            g_test = torch.zeros(n_couplings).to(device)
            with torch.no_grad():
                beta_at_zero = model(g_test).norm().item()
            print(f"    에폭 {epoch+1:3d}/{config.n_epochs}  "
                  f"손실={avg:.5f}  |β(0)|={beta_at_zero:.5f}")

    return model, losses
