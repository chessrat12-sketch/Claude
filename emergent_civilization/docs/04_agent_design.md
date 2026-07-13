# 04. 에이전트 설계 (Agent Design)

> 모든 NPC는 각각 독립적인 AI 에이전트다. 각자 자신의 기억·성격·목표·
> 인벤토리·인간관계를 개별적으로 가진다. **직업 필드는 존재하지 않는다.**

## 상태 구성 (`sim/agent.py`)

```
Agent
├─ 정체성   id, name, born_tick
├─ 위치     pos (x, y)
├─ 신체     hunger, energy, health, alive, died_tick
├─ 소유     inventory: {Resource: int}
├─ 성격     personality: Personality(greed, sociability, caution, curiosity)
├─ 목표     goal: str                (기본 "survive", LLM이 재정의 가능)
├─ 관계     relationships: {agent_id: trust ∈ [-1, 1]}
└─ 기억     memory: deque[MemoryEvent]  (최근 64개, 자동 롤오버)
```

### 왜 `job` 필드가 없는가
직업을 필드로 두는 순간 그것은 사람이 만든 구조가 된다. 대신 역할은
`metrics.role_specialization()` 에서 **행동 이력의 최빈값**으로 *관찰*된다.
"나무꾼"은 코드가 아니라 "이 에이전트가 계속 wood 를 gather 했다"는 사실이다.

## 성격 (Personality)

`[0, 1]` 범위의 4개 성향. 행동을 **강제하지 않고 편향**한다.

| 성향 | 낮음 ↔ 높음 | 영향 지점 |
|------|-------------|-----------|
| `greed` | 나눔 ↔ 축적 | 거래 가격, 공유 여부(v0.3+) |
| `sociability` | 고립 ↔ 교류 | 타 에이전트 접근, 대화 빈도 |
| `caution` | 모험 ↔ 안전 | 식량 비축 마진, 위험 회피 |
| `curiosity` | 활용 ↔ 탐험 | 미탐사 이동 vs 기존 자원 |

- 베이스라인(`HeuristicPolicy`)에서는 `caution`→비축량, `curiosity`→탐험
  방향에 직접 반영된다.
- LLM 정책에서는 프롬프트 페르소나로 주입되어 **경향**으로만 작동한다.

## 기억 (Memory)

- `MemoryEvent(tick, kind, detail)` 의 링버퍼(최근 64). 행동 결과·사망 등이
  자동 기록된다(`agent.remember`, `actions.execute` 에서 호출).
- 관찰에는 최근 6개만 노출(`observation.recent_memory`) — 컨텍스트 길이 관리.
- **확장(v0.6)**: 요약 계층 도입. 오래된 사건을 LLM으로 압축한
  "장기 기억/평판 요약"을 별도 필드로 유지하여 100+ 규모, 장기 실행에서
  토큰 폭증을 막는다.

## 관계와 신뢰 (Relationships)

- `relationships[other_id]` = 신뢰 점수 ∈ [-1, 1]. 처음부터 필드로 존재하지만
  갱신 로직은 v0.6에서 배선한다(약속 이행 → 상승, 배신 → 하락).
- 신뢰는 관찰에 노출되어(`nearby_agents[].trust`) 상대 선택에 영향을 줄 수 있다.
- 동맹·갈등은 별도 구조가 아니라 신뢰 점수의 군집/양극화로 *관찰*한다.

## 독립성 보장

각 에이전트는:
- 자신만의 `Policy` 인스턴스를 가진다(`engine` 의 `policy_factory`).
- LLM 정책일 경우 **독립 컨텍스트**로 추론한다(공유 상태 없음).
- 오직 부분 관찰을 통해서만 세계와 타 에이전트를 인지한다.

이 독립성이 "다중 에이전트 상호작용에서의 창발"이라는 연구 주장을 성립시킨다.
전역 상태를 공유하면 그것은 하나의 정책이 여러 몸을 움직이는 것에 불과하다.
