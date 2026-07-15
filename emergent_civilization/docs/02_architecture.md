# 02. 시스템 아키텍처 (Architecture)

## 3-계층 구조

```
┌────────────────────────────┐     사람이 만드는 것: 시각화 / 상호작용
│   Unity 3D  (Client)       │     - 3D 맵, NPC 렌더링, 카메라
│   - 월드/에이전트 시각화    │     - 상태를 "보여줄" 뿐, 결정하지 않음
└──────────────┬─────────────┘
               │  WebSocket / gRPC  (world state ↕ actions)
┌──────────────┴─────────────┐     사람이 만드는 것: 규칙(substrate)
│   Python  (Simulation)     │     ← 이 저장소의 `sim/`
│   - World / Agent / Rules  │     - 세계, 물질대사, 행동 실행, 지표
│   - tick loop / metrics    │     - 직업·경제·법은 없음
└──────────────┬─────────────┘
               │  LLMBackend.complete(prompt) -> action(JSON)
┌──────────────┴─────────────┐     AI가 결정하는 것
│  LLM 추론 서버              │     - 무엇을/누구와/어떤 값으로
│  (RunPod vLLM / NVIDIA API) │     - 각 에이전트 = 독립 컨텍스트
└────────────────────────────┘
```

## 왜 Python이 진실의 원천(source of truth)인가

- 결정론적 재현성(시드), 규칙의 단일 소유권, LLM 없이도 구동 가능한
  베이스라인 — 실험 신뢰성을 위해 **모든 규칙은 Python에만** 존재한다.
- Unity는 순수 프레젠테이션 계층이다. Unity를 꺼도 실험은 성립한다
  (본 저장소의 `examples/run_v01_survival.py` 가 그 증거).

## 시뮬레이션 코어 모듈 지도 (`sim/`)

| 모듈 | 책임 | 결정하지 않는 것 |
|------|------|------------------|
| `types.py` | 자원·방향·행동 어휘(Enum), `Action`/`ActionResult` | — |
| `world.py` | 그리드, `ResourceNode` 재생, 자원 총량 | — |
| `agent.py` | 신체(hunger/energy/health), 인벤토리, 기억, 관계, 성격 | 무엇을 할지 |
| `actions.py` | 행동의 **실행과 허용 여부**(rules engine) | 무엇을 할지 |
| `observation.py` | 부분적·자기중심적 관찰 생성 | — |
| `policy.py` | 관찰 → 행동. `Heuristic`(대조군) / `LLM`(실험군) | 규칙 |
| `llm_client.py` | LLM 백엔드 추상화(`Echo`/`OpenAICompat`) | 규칙 |
| `metrics.py` | 창발 지표 수집·분석 | — |
| `engine.py` | 틱 루프: 물질대사 → 결정·실행 → 세계 → 측정 | 무엇을 할지 |

**핵심 경계**: `actions.py` 는 행동이 *무엇을 하는지*를 정하고,
`policy.py` 는 *어떤 행동을 할지*를 정한다. 전자는 사람, 후자는 AI.

## 한 틱(tick)의 생애

```
for each tick:
  1. metabolize   모든 에이전트 hunger↑ energy↓ (→ 사망 가능)
  2. decide+act   생존 에이전트를 무작위 순서로:
                    obs = build_observation(부분 시야)
                    action = policy.decide(obs)      ← LLM 호출 지점
                    result = executor.execute(action) ← 규칙 적용
  3. world.step   자원 재생
  4. measure      인구/행동/지표 기록
```

무작위 순서(engine의 `shuffle`)는 특정 에이전트의 영구적 선행 이득을 없앤다.

## LLM 추론 서버 연동

`LLMBackend` 프로토콜은 `complete(prompt) -> str` 하나뿐이다.

- **개발/CI**: `EchoBackend` — 네트워크 없이 유효한 JSON 반환(오프라인).
- **실험**: `OpenAICompatBackend` — RunPod vLLM 및 NVIDIA API Catalog가
  공유하는 OpenAI `/chat/completions` 스키마 사용. `EC_LLM_BASE_URL`,
  `EC_LLM_MODEL`, `EC_LLM_API_KEY` 환경변수로 설정.

100명 규모에서는 틱당 100회 추론이 병목이므로 **동시 호출**을 지원한다
(`Simulation(max_concurrency=N)`, 서버는 `--llm-concurrency`). 공유 무료
API는 동시 요청 여유가 없어 순차 페이싱(`--llm-gap-ms`)을 쓰고, RunPod처럼
직접 소유한 vLLM 엔드포인트는 배치 처리가 가능하므로 동시성을 켠다 —
20개 동시 요청이면 100명이 5명 순차 호출한 정도의 시간에 결정을 마친다.
동시 모드에서는 그 틱 시작 시점 세계 상태를 기준으로 모두 동시에
결정하고, 행동은 이후 순서대로(셔플된 순서) 적용된다(→ `07_roadmap.md`
확장성 항목).

## 데이터 흐름 계약 (Unity ↔ Python)

- **Python → Unity**: 틱마다 월드 스냅샷(에이전트 위치·상태, 노드 잔량,
  최근 행동). 관찰과 달리 이것은 *전역* 뷰이며 렌더링 전용이다.
- **Unity → Python**: 관찰자(연구자) 입력만 — 카메라, 일시정지, 파라미터
  조정. Unity는 에이전트 행동을 만들지 않는다.
