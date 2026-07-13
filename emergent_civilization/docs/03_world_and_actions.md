# 03. 세계와 행동 어휘 (World & Action Space)

> 사람이 만드는 것: **세계, 자원, 생존 규칙, 행동 가능 목록**.
> 이 문서는 그 substrate 의 사양이다. 이 밖의 것(직업·가격·법)은 정의하지 않는다.

## 세계 (World)

- 유한 2D 그리드 (`width × height`), 가장자리는 이동 차단(wrap 없음).
- 각 타일은 최대 1개의 `ResourceNode` 를 가질 수 있다.
- `make_scattered_world(width, height, density, seed)` 로 자원을 무작위 배치.
  식량이 과대표집되어 순수 생존은 가능하되 자명하지 않게 튜닝.

## 자원 (Resource)

| 자원 | 용도(현재) | 용도(로드맵) |
|------|-----------|--------------|
| `FOOD` | 섭취 → 허기 감소, 생존 | 거래 대상, 가격 형성 |
| `WOOD` | 축적 | 제작 재료(v0.4), 거래 |
| `STONE` | 축적 | 제작 재료(v0.4), 거래 |

`ResourceNode` 는 채집으로 고갈되고 `regen_per_tick` 으로 서서히 재생(용량 상한).
식량 재생이 목재/석재보다 빠르다 → 물질은 상대적으로 희소.

## 생존 규칙 (Survival, `agent.py` 상수)

| 상수 | 값 | 의미 |
|------|----|------|
| `HUNGER_PER_TICK` | 4 | 매 틱 허기 증가 |
| `ENERGY_PER_TICK` | 2 | 매 틱 에너지 감소 |
| `STARVING_HUNGER` | 90 | 이 이상이면 체력 감소 |
| `EXHAUSTED_ENERGY` | 10 | 이 이하이면 체력 감소 |
| `HEALTH_DECAY` | 6 | 굶주림/탈진 시 체력 손실/틱 |
| `HEALTH_REGEN` | 2 | 배부르고 쉰 상태에서 체력 회복/틱 |

체력이 0이면 사망(`alive=False`, `died_tick` 기록). 이 6개 숫자가 사회 형성의
압력을 만든다 — 너무 관대하면 협력 유인이 없고, 너무 가혹하면 사회 이전에 멸종한다.
튜닝은 "베이스라인으로 일부는 살고 일부는 죽는" 지점을 목표로 한다.

## 행동 어휘 (Action Space, `types.ActionType`)

행동은 전체 어휘를 **처음부터 모두 노출**하되, 현재 마일스톤 밖의 행동은
파싱은 되지만 실행 시 명확한 실패 사유와 함께 거부된다(`actions.IMPLEMENTED_ACTIONS`).
이렇게 하면 에이전트가 "가능의 경계"를 스스로 발견한다.

| 행동 | args | 도입 | 효과 | 상태 |
|------|------|------|------|------|
| `move` | `{direction}` | v0.1 | 인접 타일로 이동 | ✅ |
| `eat` | — | v0.1 | FOOD 1 소비 → 허기 −35 | ✅ |
| `rest` | — | v0.1 | 에너지 +30 (은신처 안 +15 추가) | ✅ |
| `idle` | — | v0.1 | 아무것도 안 함 | ✅ |
| `gather` | — | v0.2 | 현재 타일 노드 채집(도구 보유 시 +2) | ✅ |
| `speak` | `{to, message}` | v0.3 | 인접 에이전트에게 메시지 | ✅ |
| `trade` | `{to, give, receive}` 또는 `{offer, accept}` | v0.3 | 교환 제안/응답 | ✅ |
| `give` | `{to, items}` | v0.4 | 대가 없이 자원 전달(신뢰 +0.2) | ✅ |
| `craft` | — | v0.4 | 2 wood + 1 stone → 도구(TOOL) | ✅ |
| `build` | — | v0.4 | 3 wood + 1 stone → 은신처 | ✅ |
| `propose_rule` | `{text}` | v0.7 | 인접 에이전트에게 규범 제안 | ⬜ |

추가 substrate: **밤낮 주기**(`World.day_length`)와 **포식자**(밤에만),
**은신처**(`Structure`), 크래프트 아이템 **TOOL** — 상세는 `09_ecology.md`.

### 설계 원칙
- **가능성만 열고 강제하지 않는다**: `trade`/`speak`/`propose_rule` 은
  존재하지만, 언제·누구와 쓸지는 전적으로 AI의 결정이다.
- **사회 구조 어휘 없음**: `hire`, `set_price`, `make_law` 같은 상위 개념
  행동은 두지 않는다. 그런 것들은 기본 행동의 *패턴*으로 창발해야 한다
  (예: "가격"은 반복된 `trade` 의 교환비 수렴으로 관찰).

## 확장 지침

새 마일스톤을 열 때:
1. `IMPLEMENTED_ACTIONS` 에 행동 추가 + `ActionExecutor` 에 핸들러 구현.
2. 필요한 관찰 필드를 `observation.py` 에 추가(여전히 부분 시야 유지).
3. 대응하는 지표를 `metrics.py` 에 배선(카운터는 이미 존재).
4. 프롬프트에는 **행동의 존재**만 알리고 사용법은 예시로 강제하지 않는다.
