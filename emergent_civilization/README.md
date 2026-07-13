# Emergent Civilization

**LLM 기반 자율 AI 문명 시뮬레이션**

> 최소한의 규칙만 제공된 환경에서 LLM 기반 다중 AI 에이전트는
> 자발적으로 사회를 형성할 수 있는가?

여러 개의 독립적인 LLM 에이전트를 하나의 환경에 배치하고, 사람의 직접적인
규칙 설계 없이 직업·경제·법·협력 같은 사회 구조가 **스스로 창발**하는지를 관찰한다.

## 핵심 원칙

| 사람이 만드는 것 (substrate) | AI가 결정하는 것 |
|------------------------------|------------------|
| 세계(맵), 자원, 생존 규칙, 행동 가능 목록 | 무엇을·누구와·어떤 값에·누구를 신뢰할지·규칙 제안 여부 |

이 저장소의 `sim/` 은 세계·물질대사·행동 어휘만 정의하며, **직업·가격·법·
동맹을 표현하는 코드는 없다**. 그런 구조는 에이전트 행동의 패턴으로만 관찰된다.

## 현재 상태

- ✅ **v0.1 생존 / v0.2 채집** — substrate + 대조군(`HeuristicPolicy`) +
  실험군(`LLMPolicy`, 백엔드 교체식) 구현.
- ⏳ **v0.3 거래** — 다음 단계 (→ `docs/07_roadmap.md`).

오프라인 베이스라인에서 10명이 200틱 전원 생존, 부의 지니 ≈ 0.18 → 세계가
"생존 가능(solvable)"함을 검증.

## 빠른 시작

```bash
# 의존성 없음 — 순수 표준 라이브러리로 오프라인 실행
cd emergent_civilization
python -m examples.run_v01_survival     # 휴리스틱 베이스라인 데모
python tests/test_survival.py           # 테스트 (또는: pytest tests/)
```

### LLM 에이전트로 실행하려면

`OpenAICompatBackend` 를 `LLMPolicy` 에 연결한다(RunPod vLLM / NVIDIA API 호환):

```bash
export EC_LLM_BASE_URL="https://<your-endpoint>/v1"
export EC_LLM_MODEL="<model-name>"
export EC_LLM_API_KEY="<key>"
```

```python
from sim import Simulation, LLMPolicy, make_scattered_world
from sim.llm_client import OpenAICompatBackend

backend = OpenAICompatBackend()   # 위 환경변수 사용
sim = Simulation(world, agents,
                 policy_factory=lambda a: LLMPolicy(a, backend))
```

## 구조

```
emergent_civilization/
├── sim/                     # 시뮬레이션 코어 (사람이 만든 substrate)
│   ├── types.py             # 자원·방향·행동 어휘
│   ├── world.py             # 그리드, 자원 노드, 재생
│   ├── agent.py             # 신체·기억·성격·관계 (직업 없음)
│   ├── actions.py           # 행동 실행·허용 (rules engine)
│   ├── observation.py       # 부분·자기중심 관찰
│   ├── policy.py            # Heuristic(대조) / LLM(실험)
│   ├── llm_client.py        # LLM 백엔드 (Echo / OpenAICompat)
│   ├── metrics.py           # 창발 지표 수집·분석
│   └── engine.py            # 틱 루프
├── examples/
│   └── run_v01_survival.py  # 오프라인 생존 데모
├── tests/
│   └── test_survival.py
└── docs/
    ├── 01_research_design.md          # 연구 질문·가설·방법론
    ├── 02_architecture.md             # Unity + Python + LLM 3-계층
    ├── 03_world_and_actions.md        # 세계·자원·생존 규칙·행동 어휘
    ├── 04_agent_design.md             # 에이전트 상태·성격·기억·관계
    ├── 05_prompt_and_decision_loop.md # 프롬프트 설계·편향 통제
    ├── 06_metrics.md                  # 측정 항목
    └── 07_roadmap.md                  # v0.1 → v1.0 로드맵
```

## 실험 환경

- Unity 기반 3D 프레젠테이션 (렌더링 전용, 결정하지 않음)
- Python 시뮬레이션 서버 (진실의 원천 — 이 저장소)
- LLM 추론 서버 (RunPod vLLM 또는 NVIDIA API)
- NPC 수: 초기 10명 → 확장 목표 100명 이상

자세한 설계 근거는 `docs/` 참조.
