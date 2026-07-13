# 08. 시각화 (Visualization)

> 세계를 "작은 마을"처럼 3D로 본다. **시각화는 시뮬레이션을 바꾸지 않는다** —
> 렌더 전용이며, 에이전트는 이 전역 뷰를 결코 보지 못한다(→ `sim/snapshot.py`).

## 한 개의 계약, 두 개의 클라이언트

```
        ┌──────────────────────────┐
        │  Python  viz_server.py    │  백그라운드 스레드가 sim.step()
        │  - Simulation 구동         │  틱마다 스냅샷 갱신
        │  - GET /state (JSON)      │
        └───────────┬──────────────┘
        GET /state  │  (동일 스냅샷)
        ┌───────────┴───────────┐
        ▼                       ▼
  viewer/village.html      unity/ (C# 클라이언트)
  무설치 브라우저 뷰어       3D 작은 마을
  아이소메트릭 캔버스        UnityWebRequest + JsonUtility
```

렌더 스냅샷은 `sim/snapshot.py: world_snapshot()` 하나가 만든다. 브라우저와
Unity가 **정확히 같은 JSON**을 소비하므로 두 화면은 항상 같은 세계를 보여준다.

## 스냅샷 vs 관찰 (중요한 구분)

| | 관찰(observation) | 스냅샷(snapshot) |
|---|---|---|
| 대상 | 에이전트 | 렌더러 |
| 시야 | 부분·자기중심 | 전역(god's-eye) |
| 용도 | **의사결정 입력** | **렌더링 전용** |
| 위험 | — | 에이전트에 주면 전역 지식 누출 → 절대 금지 |

스냅샷은 오직 서버가 렌더 클라이언트에 보낼 뿐, 정책에 주입되지 않는다.

## 실행

```bash
cd emergent_civilization
python -m server.viz_server                 # http://localhost:8000 (브라우저 자동 서빙)
python -m server.viz_server --agents 20 --size 18 --tick-ms 250 --seed 5
```

- 브라우저: `http://localhost:8000` 접속 → 아이소메트릭 마을 + HUD.
- Unity: `unity/README_UNITY.md` 5분 설정 → 같은 서버에 연결.
- 다른 클라이언트: `GET /state` 로 JSON 스냅샷 폴링.

## 스냅샷 스키마 (Unity `JsonUtility` 호환)

맵(dict)을 쓰지 않고 배열·스칼라만 사용한다(인벤토리는 `{resource,amount}` 리스트).

```jsonc
{
  "tick": 58, "width": 14, "height": 14,
  "nodes":  [{"x":3,"y":5,"resource":"food","amount":8,"capacity":12}],
  "agents": [{"id":"a0","name":"Aria","x":3,"y":5,"hunger":40,"energy":70,
              "health":100,"alive":true,"wealth":5,"lastAction":"gather",
              "inventory":[{"resource":"food","amount":3}]}],
  "events": [{"type":"trade","a":"a0","b":"a1","text":"Aria ⇄ Boaz: 2 wood for 1 food"}],
  "stats":  {"tick":58,"alive":12,"population":12,"deaths":0,"trades":6,
             "messages":0,"gini":0.30,"priceWoodForFood":0.5}
}
```

## 브라우저 뷰어 (`viewer/village.html`)

- 순수 캔버스, 외부 의존성 0. `/state` 를 300ms 폴링, 프레임 간 위치 보간으로
  에이전트가 부드럽게 이동.
- 렌더: 아이소메트릭 잔디 타일(약간의 두께로 3D감), 식량=열매 덤불, 나무=수목,
  돌=바위, 에이전트=이름·체력바를 가진 작은 사람, 거래/대화 시 두 에이전트를 잇는 펄스.
- HUD: 틱·생존·거래·대화·지니·창발 가격 + 사건 피드.

## Unity 클라이언트 (`unity/`)

- `SnapshotModels.cs`(직렬화), `SimClient.cs`(폴링), `VillageRenderer.cs`(렌더).
- 프리팹 미지정 시 프리미티브로 즉시 마을 구성(캡슐/실린더/구/플레인).
- 에이전트는 `glideSpeed` 로 새 칸으로 이동, 노드 프롭은 잔량에 비례해 크기 변화.

## 확장

- 이벤트 로그를 JSONL 로 저장하는 `/events` 엔드포인트(사후 분석 파이프라인).
- 100+ 규모: 스냅샷 델타 전송, WebSocket 푸시, 뷰포트 컬링.
- LLM 실행 시각화: 서버 policy_factory 만 `LLMPolicy` 로 교체(클라이언트 무변경).
