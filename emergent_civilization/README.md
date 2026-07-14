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

- ✅ **v0.1–v0.4**: 생존 · 채집 · 거래/대화 · 분업/제작.
- ✅ **살아있는 생태계**: 밤/포식자(위협), 은신처(주거), 도구(제작),
  선물·신뢰(인간관계) — 서로 맞물린 유인 구조. → `docs/09_ecology.md`
- ✅ **3D 시각화**: Python 서버가 월드 스냅샷을 스트리밍. **진짜 3D 브라우저
  뷰어**(Three.js, 궤도 카메라, 설치 불필요, 기본 `/`)· 가벼운 아이소메트릭
  뷰(`/iso`) · Unity 3D(원클릭 부트스트랩)가 같은 계약을 소비.
- ✅ **LLM 데모**: 관찰→프롬프트→모델→행동 배선과 `reason` 로그.
  RunPod/NVIDIA/Anthropic 또는 오프라인 Mock 백엔드.

핵심은 세계에 **위험과 필요**만 넣었다는 것이다 — "뭉쳐라·나눠라·거래하라"는
지시는 없다. 베이스라인 200틱에서 은신처 6·도구 5·선물 6·동맹 1·평균신뢰
0.43, 사망 5(포식자 4·기아 1): **위협 앞에서 협력이 창발**한다.

## 빠른 시작

```bash
cd emergent_civilization

# 1) 진짜 3D로 마을 보기 — 서버 실행 후 브라우저에서 http://localhost:8000
python -m server.viz_server                 # 궤도 카메라(좌드래그 회전·휠 줌)
                                             # 가벼운 2D 버전은 /iso

# 2) LLM 에이전트 추론 데모 (오프라인 Mock, 또는 env로 실제 모델)
python -m examples.run_llm_agents

# 3) 헤드리스 베이스라인 데모 + 테스트 (의존성 0, 순수 표준 라이브러리)
python -m examples.run_v01_survival
python tests/test_survival.py               # 15개 (또는: pytest tests/)
```

실제 LLM 연결:
```bash
# OpenAI 호환(RunPod vLLM / NVIDIA API)
export EC_LLM_BASE_URL=... EC_LLM_MODEL=... EC_LLM_API_KEY=...
# 또는 Anthropic
export EC_LLM_API_KEY=<anthropic-key> EC_LLM_MODEL=claude-haiku-4-5-20251001
python -m examples.run_llm_agents
```

Unity 3D 클라이언트 설정은 `unity/README_UNITY.md` (5분), 시각화 구조는
`docs/08_visualization.md` 참조.

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
│   ├── interactions.py      # 메시지·거래 제안 전송 계층 (v0.3)
│   ├── ecology.py           # 밤·노출·포식자 (위협)
│   ├── observation.py       # 부분·자기중심 관찰
│   ├── policy.py            # Heuristic(대조) / LLM(실험)
│   ├── llm_client.py        # LLM 백엔드 (Mock/Echo/OpenAICompat/Anthropic)
│   ├── metrics.py           # 창발 지표 수집·분석
│   ├── snapshot.py          # 전역 렌더 스냅샷 (시각화 전용)
│   └── engine.py            # 틱 루프
├── server/
│   └── viz_server.py        # 라이브 시뮬레이션 + GET /state 스트리밍
├── viewer/
│   ├── village_3d.html       # 진짜 3D(Three.js) 뷰어 — 기본(/), 궤도 카메라
│   ├── village.html          # 무설치 아이소메트릭 뷰어 — /iso
│   └── vendor/                # 번들된 Three.js(CDN 미사용, 오프라인 동작)
├── unity/                   # Unity 3D 클라이언트 (C# + 설정 가이드)
│   ├── Scripts/*.cs
│   └── README_UNITY.md
├── examples/
│   ├── run_v01_survival.py  # 오프라인 베이스라인 데모(생태계 포함)
│   └── run_llm_agents.py    # LLM 에이전트 추론(reason) 데모
├── tests/
│   └── test_survival.py
└── docs/
    ├── 01_research_design.md          # 연구 질문·가설·방법론
    ├── 02_architecture.md             # Unity + Python + LLM 3-계층
    ├── 03_world_and_actions.md        # 세계·자원·생존 규칙·행동 어휘
    ├── 04_agent_design.md             # 에이전트 상태·성격·기억·관계
    ├── 05_prompt_and_decision_loop.md # 프롬프트 설계·편향 통제
    ├── 06_metrics.md                  # 측정 항목
    ├── 07_roadmap.md                  # v0.1 → v1.0 로드맵
    ├── 08_visualization.md            # 시각화 (브라우저 + Unity)
    └── 09_ecology.md                  # 생태계: 위협·주거·인간관계
```

## 실험 환경

- Unity 기반 3D 프레젠테이션 (렌더링 전용, 결정하지 않음)
- Python 시뮬레이션 서버 (진실의 원천 — 이 저장소)
- LLM 추론 서버 (RunPod vLLM 또는 NVIDIA API)
- NPC 수: 초기 10명 → 확장 목표 100명 이상

자세한 설계 근거는 `docs/` 참조.
