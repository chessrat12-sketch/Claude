# AdS/CFT 홀로그래피 + 머신러닝

AdS/CFT 대응과 머신러닝의 수학적 구조를 탐구하는 Python 프로젝트.

## 핵심 물리

### AdS/CFT 대응 (Maldacena 1997)
```
d차원 CFT (경계) ↔ (d+1)차원 AdS 중력 (벌크)
```

| CFT (경계) | AdS (벌크) |
|-----------|-----------|
| 연산자 소스 J(x) | 비정규화 모드 φ⁽⁰⁾ |
| 1점 함수 ⟨O(x)⟩ | 정규화 모드 φ⁽¹⁾ |
| 연산자 차원 Δ | 벌크 질량 m²L²=Δ(Δ-d) |
| 중심 전하 c | c = 3L/(2G_N) |
| 얽힘 엔트로피 S | RT 측지선 길이 / 4G_N |
| 열 상태 (온도 T) | BTZ 블랙홀 |
| RG 흐름 | 홀로그래픽 radial 방향 z |

### Ryu-Takayanagi 공식 (2006)
```
S(A) = Length(γ_A) / (4G_N)
```
AdS₃/CFT₂에서 `γ_A`는 반원 측지선, 길이 = `2L log(l/ε)`.

CFT 결과와 정확히 일치: `S = (c/3) log(l/ε)`.

### 심층 학습 ↔ 홀로그래픽 RG (Hashimoto et al. 2018)

| 신경망 | 홀로그래픽 RG |
|--------|-------------|
| 층 인덱스 t | AdS 지름 방향 z |
| 활성화 벡터 h_t | 벌크 장 φ(z,x) |
| 입력층 | UV 경계 (z→0) |
| 출력층 | IR 고정점 |
| β 함수 학습 | RG 흐름 방정식 |
| 손실 함수 | 온쉘 AdS 작용 |

### MERA ↔ AdS (Swingle 2012)

| MERA | AdS 기하학 |
|------|-----------|
| 층 k | z = L·2^k/N |
| disentangler | RT 측지선 절단 |
| isometry | 벌크-경계 전파자 |
| 인과 원뿔 | entanglement wedge |

## 구조

```
ads_cft_ml/
├── geometry.py        # AdS₃ 기하학, 측지선, 계량
├── entanglement.py    # Ryu-Takayanagi, Calabrese-Cardy
├── cft.py             # 상관함수, Virasoro, c-정리
└── tensor_network.py  # MERA, Swingle 대응

ml/
├── holographic_rg.py  # 신경망 = RG 흐름
└── bulk_boundary.py   # HKLL 재건, 벌크-경계 NN

examples/
├── 01_geodesics_rt.py              # RT 공식 검증
├── 02_entanglement_phase_transition.py  # 홀로그래픽 위상 전이
├── 03_neural_rg_flow.py            # 신경망 ↔ RG 흐름
└── 04_mera_holography.py           # MERA ↔ AdS 기하학
```

## 실행

```bash
pip install numpy scipy

# 예제 실행
python -m examples.01_geodesics_rt
python -m examples.02_entanglement_phase_transition
python -m examples.03_neural_rg_flow
python -m examples.04_mera_holography
```

## 참고 문헌

- Maldacena (1997): "The large N limit of superconformal field theories and supergravity"
- Ryu & Takayanagi (2006): "Holographic derivation of entanglement entropy"
- Swingle (2012): "Entanglement renormalization and holography"
- Hashimoto, Iizuka & Nishida (2018): "Deep learning and the AdS/CFT correspondence"
- Almheiri et al. (2019): "The entropy of bulk quantum fields and the entanglement wedge"
- Penington (2020): "Entanglement wedge reconstruction and the information paradox"
