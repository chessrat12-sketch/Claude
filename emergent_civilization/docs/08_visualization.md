# 08. 시각화 (Visualization)

> 세계를 "작은 마을"처럼 3D로 본다. **시각화는 시뮬레이션을 바꾸지 않는다** —
> 렌더 전용이며, 에이전트는 이 전역 뷰를 결코 보지 못한다(→ `sim/snapshot.py`).

## 한 개의 계약, 세 개의 클라이언트

```
        ┌──────────────────────────┐
        │  Python  viz_server.py    │  백그라운드 스레드가 sim.step()
        │  - Simulation 구동         │  틱마다 스냅샷 갱신
        │  - GET /state (JSON)      │
        └───────────┬──────────────┘
        GET /state  │  (동일 스냅샷)
        ┌───────────┼───────────────┐
        ▼            ▼               ▼
  village_3d.html  village.html    unity/ (C# 클라이언트)
  진짜 3D(Three.js) 아이소메트릭     Unity 엔진 3D
  무설치·궤도 카메라  무설치 캔버스   UnityWebRequest + JsonUtility
     "/" (기본)         "/iso"        SceneBootstrap 원클릭
```

렌더 스냅샷은 `sim/snapshot.py: world_snapshot()` 하나가 만든다. 세 클라이언트
모두 **정확히 같은 JSON**을 소비하므로 항상 같은 세계를 보여준다.

## 설치 없이 진짜 3D 보기 (권장 시작점)

```bash
cd emergent_civilization
python -m server.viz_server        # 브라우저에서 http://localhost:8000 (기본이 3D)
```

`viewer/village_3d.html` 은 [Three.js](https://threejs.org/)로 실제 원근
3D 씬을 그린다 — 좌드래그 회전·휠 줌·우드래그 이동이 되는 진짜 궤도 카메라.
CDN에 의존하지 않도록 Three.js를 `viewer/vendor/`에 **번들**해 뒀다
(`server/viz_server.py` 가 `/vendor/*` 로 서빙) — 사내망·오프라인에서도 동작.
아이소메트릭 2D 캔버스 버전은 `/iso` 로 유지된다(더 가볍고 즉시 로드).

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

- 브라우저(진짜 3D): `http://localhost:8000/` 또는 `/3d` → Three.js 궤도 카메라.
- 브라우저(아이소메트릭): `http://localhost:8000/iso` → 가벼운 2D 캔버스.
- Unity: `unity/README_UNITY.md` 의 `SceneBootstrap` 원클릭 → 같은 서버에 연결.
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

## 3D 브라우저 뷰어 (`viewer/village_3d.html`) — 기본(`/`)

- **Three.js**(`viewer/vendor/`에 번들, CDN 미사용) + `OrbitControls` — 진짜
  원근 카메라로 회전·줌·이동 가능. `/state` 300ms 폴링, `lerp` 보간 이동.
- 렌더: 지면 + 그리드, 나무(원뿔+원기둥)/바위(정이십면체)/은신처(상자+지붕)/
  포식자(캡슐+빛나는 눈)/에이전트(캡슐+구, 이름표·체력바 스프라이트).
- 낮/밤: `HemisphereLight`+`DirectionalLight` 강도와 하늘색·안개가 전환.
- 이벤트: 물릴 때 붉은 emissive 플래시, 사건 피드에 거래/선물/건축/공격 로그.
- HUD는 아이소메트릭 뷰와 동일 지표(생존·은신처·도구·거래·선물·동맹·평균신뢰 등).

## 아이소메트릭 뷰어 (`viewer/village.html`) — `/iso`

- 순수 캔버스, 외부 의존성 0(가장 가벼움, 즉시 로드). `/state` 300ms 폴링.
- 렌더: 아이소메트릭 잔디 타일(약간의 두께로 3D감), 식량=열매 덤불, 나무=수목,
  돌=바위, 은신처=오두막, 포식자=늑대, 거래/선물/대화 시 두 에이전트를 잇는 펄스.

## Unity 클라이언트 (`unity/`)

- `SceneBootstrap.cs` 하나로 카메라(+`CameraOrbit.cs`)·조명·지면·렌더러·
  클라이언트를 실행 시점에 전부 생성·연결(수동 GameObject 배치 불필요).
- `SnapshotModels.cs`(직렬화), `SimClient.cs`(폴링), `VillageRenderer.cs`(렌더:
  프리팹 미지정 시 프리미티브로 즉시 구성, 낮/밤 조명, 체력바, 이벤트 팝업).
- 에이전트는 `glideSpeed` 로 새 칸으로 이동, 노드 프롭은 잔량에 비례해 크기 변화.

## 3D 뷰어를 CDN 대신 로컬에 번들한 이유

`village_3d.html` 은 처음엔 unpkg CDN에서 Three.js를 불러왔으나, 사내망·방화벽
·오프라인 환경에서 깨질 수 있어 `npm pack three` 로 받은
`three.module.min.js`/`OrbitControls.js` 를 `viewer/vendor/`에 커밋하고
서버가 `/vendor/*` 로 직접 서빙하도록 바꿨다. 결과적으로 **인터넷 연결 없이도**
3D 뷰어가 동작한다(라이선스: `viewer/vendor/THREE_LICENSE.txt`, MIT).

## 확장

- 이벤트 로그를 JSONL 로 저장하는 `/events` 엔드포인트(사후 분석 파이프라인).
- 100+ 규모: 스냅샷 델타 전송, WebSocket 푸시, 뷰포트 컬링.
- LLM 실행 시각화: 서버 policy_factory 만 `LLMPolicy` 로 교체(클라이언트 무변경).
