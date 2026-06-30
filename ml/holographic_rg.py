"""
홀로그래픽 RG 흐름으로서의 심층 신경망

이론적 배경:
  - Wilsonian RG: UV 자유도를 적분하여 유효 이론 생성
  - 홀로그래픽 RG: AdS 반지름 방향 z 따라 Hamilton-Jacobi 방정식
  - 심층 학습: 층 t에서 표현을 변환 → RG 스텝

핵심 방정식:
  RG 흐름:       dg_i/d(log μ) = β_i(g)
  홀로그래픽:    dφ/dz = F[φ]    (EOM in AdS)
  신경망:         h_{t+1} = σ(W_t h_t + b_t)

β 함수를 신경망으로 학습하여 RG 흐름 궤적 재현.
"""
import numpy as np
from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass, field


# ── Numpy 기반 간단한 신경망 (PyTorch 없이도 동작) ─────────────

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)

def tanh_grad(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2


@dataclass
class RGLayer:
    """
    한 번의 RG 스텝 = 한 신경망 층.
    z 방향으로 dz 만큼 전파.
    """
    W: np.ndarray      # 가중치 행렬 (dim × dim)
    b: np.ndarray      # 편향 벡터 (dim,)
    z: float           # 현재 AdS z 좌표
    dz: float          # RG 스텝 크기
    activation: str = "tanh"

    def __post_init__(self):
        self._act = {"tanh": tanh, "relu": relu, "sigmoid": sigmoid}[self.activation]
        self._act_grad = {"tanh": tanh_grad, "relu": relu_grad,
                          "sigmoid": lambda x: sigmoid(x) * (1 - sigmoid(x))}[self.activation]

    def forward(self, phi: np.ndarray) -> np.ndarray:
        """φ(z) → φ(z + dz): 한 RG 스텝."""
        return self._act(self.W @ phi + self.b)

    def beta_function(self, phi: np.ndarray) -> np.ndarray:
        """
        β 함수: dφ/d(log z) = β(φ)
        근사: β(φ) ≈ (forward(φ) - φ) / dz * z
        """
        return (self.forward(phi) - phi) * (self.z / self.dz)


class HolographicRGNetwork:
    """
    홀로그래픽 RG를 구현하는 심층 신경망.

    구조:
      입력: UV 경계 데이터 φ_UV (z = z_min)
      은닉층: z 방향 전파 (각 층 = 한 RG 스텝)
      출력: IR 고정점 근방 데이터 φ_IR (z = z_max)

    각 층의 AdS 계량에서:
      가중치 W_t ~ 1/z_t (계량 인수 L/z)
      스텝 크기 dz ~ z_t * Δ(log z)
    """
    def __init__(
        self,
        dim: int,
        n_layers: int,
        z_uv: float = 0.01,
        z_ir: float = 10.0,
        activation: str = "tanh",
        seed: int = 42,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.z_uv = z_uv
        self.z_ir = z_ir

        rng = np.random.default_rng(seed)
        # z 값: log 간격 (AdS 기하학에서 자연스러운 척도)
        log_z = np.linspace(np.log(z_uv), np.log(z_ir), n_layers + 1)
        z_values = np.exp(log_z)
        dz_values = np.diff(z_values)

        self.layers: List[RGLayer] = []
        for k in range(n_layers):
            z_k = z_values[k]
            # AdS 계량 인수: L/z (가중치 스케일)
            scale = 1.0 / np.sqrt(dim * z_k)
            W = rng.normal(0, scale, (dim, dim))
            # 직교 초기화 (RG 흐름의 안정성)
            Q, _ = np.linalg.qr(W)
            b = rng.normal(0, 0.01, dim)
            self.layers.append(
                RGLayer(W=Q, b=b, z=z_k, dz=dz_values[k], activation=activation)
            )

    def forward(self, phi_uv: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        UV 경계 → IR: 전체 RG 흐름 실행.
        반환: (phi_ir, [phi_0, phi_1, ..., phi_n])  — 각 층의 활성화
        """
        phi = phi_uv.copy()
        trajectory = [phi.copy()]
        for layer in self.layers:
            phi = layer.forward(phi)
            trajectory.append(phi.copy())
        return phi, trajectory

    def rg_trajectory(self, phi_uv: np.ndarray) -> np.ndarray:
        """
        RG 흐름 궤적 φ(z) 반환.
        shape: (n_layers+1, dim)
        """
        _, traj = self.forward(phi_uv)
        return np.array(traj)

    def beta_function_trajectory(self, phi_uv: np.ndarray) -> np.ndarray:
        """
        각 층에서의 β 함수값 = dφ/d(log z).
        RG 흐름의 속도를 나타냄.
        """
        _, traj = self.forward(phi_uv)
        betas = []
        for k, layer in enumerate(self.layers):
            beta_k = layer.beta_function(traj[k])
            betas.append(beta_k)
        return np.array(betas)

    def c_function(self, phi_uv: np.ndarray, c_uv: float) -> np.ndarray:
        """
        Zamolodchikov c-함수의 홀로그래픽 근사.
        RG 흐름 따라 단조 감소.

        c(z) = c_UV - 12π ∫ |β(φ)|² d(log z)
        """
        betas = self.beta_function_trajectory(phi_uv)
        beta_sq = np.sum(betas ** 2, axis=-1)
        # 누적 적분 (trapezoidal)
        dlogz = (np.log(self.z_ir) - np.log(self.z_uv)) / self.n_layers
        c_decrease = 12 * np.pi * np.cumsum(beta_sq) * dlogz
        c_values = c_uv - np.concatenate([[0.0], c_decrease])
        return np.maximum(c_values, 0.0)

    def entanglement_entropy_estimate(self, phi_uv: np.ndarray) -> np.ndarray:
        """
        각 층에서의 얽힘 엔트로피 추정.
        S(z) ∝ log(표현의 상관 길이) ∝ 층의 유효 정보량
        """
        _, traj = self.forward(phi_uv)
        entropies = []
        for phi_z in traj:
            # 활성화 분포의 엔트로피 (볼츠만 해석)
            probs = np.abs(phi_z)
            probs = probs / (probs.sum() + 1e-15)
            S = -np.sum(probs * np.log(probs + 1e-15))
            entropies.append(S)
        return np.array(entropies)


# ── Wilsonian RG 학습 ──────────────────────────────────────────

class WilsonianRG:
    """
    Wilsonian RG: β 함수를 신경망으로 학습.

    목표: 주어진 CFT 데이터(상관함수 등)로부터
    β(g) = dg/d(log μ) 를 역공학.
    """
    def __init__(self, n_couplings: int, n_hidden: int = 32, seed: int = 0):
        self.n_couplings = n_couplings
        rng = np.random.default_rng(seed)

        # β 함수 네트워크: g → β(g)
        scale = np.sqrt(2.0 / n_couplings)
        self.W1 = rng.normal(0, scale, (n_hidden, n_couplings))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, scale / np.sqrt(n_hidden), (n_couplings, n_hidden))
        self.b2 = np.zeros(n_couplings)

    def beta(self, g: np.ndarray) -> np.ndarray:
        """β(g) = dg/d(log μ) 예측."""
        h = tanh(self.W1 @ g + self.b1)
        return self.W2 @ h + self.b2

    def flow(self, g0: np.ndarray, n_steps: int, dlogmu: float = 0.1) -> np.ndarray:
        """
        UV에서 IR로 RG 흐름 적분.
        dg/d(log μ) = β(g)  →  오일러 방법으로 적분
        """
        g = g0.copy()
        trajectory = [g.copy()]
        for _ in range(n_steps):
            g = g - dlogmu * self.beta(g)  # IR로: μ 감소
            trajectory.append(g.copy())
        return np.array(trajectory)

    def fixed_points(self, g0: np.ndarray, tol: float = 1e-6) -> Optional[np.ndarray]:
        """
        RG 고정점 탐색: β(g*) = 0.
        간단한 뉴턴 반복법.
        """
        g = g0.copy()
        for _ in range(1000):
            b = self.beta(g)
            if np.linalg.norm(b) < tol:
                return g
            # 단순 경도 하강
            g = g - 0.01 * b
        return g if np.linalg.norm(self.beta(g)) < 1e-4 else None


Optional = type(None).__class__.__mro__[0]  # type hint용
try:
    from typing import Optional
except ImportError:
    pass
