# Unity 3D 클라이언트 — 작은 마을 시각화

Python 시뮬레이션(`server/viz_server.py`)이 스트리밍하는 월드 스냅샷을
Unity에서 3D로 렌더링한다. **Unity는 순수 프레젠테이션 계층이며 시뮬레이션의
결정에 관여하지 않는다** — 브라우저 뷰어(`viewer/village.html`)와 정확히 같은
`GET /state` 엔드포인트를 소비한다.

## 준비물
- Unity 2021 LTS 이상 (JsonUtility·UnityWebRequest는 기본 내장, 외부 패키지 불필요)
- 실행 중인 Python 서버:
  ```bash
  cd emergent_civilization
  python -m server.viz_server        # http://localhost:8000
  ```

## 설정 (5분)

1. **새 3D 프로젝트** 생성 (URP/Built-in 무관).
2. `unity/Scripts/` 의 세 파일을 프로젝트 `Assets/` 아래로 복사:
   - `SnapshotModels.cs` — 스냅샷 JSON의 직렬화 모델
   - `SimClient.cs` — `/state` 폴링
   - `VillageRenderer.cs` — 스냅샷 → GameObject
3. 씬에 빈 GameObject `Village` 를 만들고 **VillageRenderer** 컴포넌트 추가.
4. 빈 GameObject `SimClient` 를 만들고 **SimClient** 컴포넌트 추가.
   - `Server Url` = `http://localhost:8000`
   - `Poll Interval` = 서버 `--tick-ms` 와 맞춤 (기본 0.3)
   - `Village Renderer` 슬롯에 3의 `Village` 오브젝트를 드래그.
5. 카메라를 맵을 내려다보게 배치 (예: position `(7, 14, -6)`, rotation `(55, 0, 0)`;
   14×14 월드 기준). ▶ Play.

프리팹을 지정하지 않으면 렌더러가 **프리미티브**(캡슐=에이전트, 실린더=나무,
구=돌, 플레인=지면)로 즉시 마을을 구성한다. 아트를 붙이려면 인스펙터의
`Tree Prefab / Rock Prefab / Agent Prefab` 슬롯에 프리팹을 넣으면 된다.

## 좌표 규약
- 월드 1칸 = Unity 1 유닛. 노드/에이전트는 `(x, 0.5, y)` 에 배치된다.
- 에이전트는 `VillageRenderer.glideSpeed` 로 새 칸으로 부드럽게 이동한다.
- 노드 프롭은 잔량(`amount/capacity`)에 비례해 크기가 변한다.

## 스냅샷 스키마 (`sim/snapshot.py` 와 1:1)
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
> 맵에 dictionary가 없도록 설계되어 `JsonUtility` 로 바로 파싱된다.
> `priceWoodForFood` 는 거래 전 `null` 일 수 있으며, 이때 `JsonUtility` 는 0으로 둔다.

## 확장 아이디어
- `events` 의 trade/speak 를 파티클/말풍선으로 표시 (브라우저 뷰어의 pulse 참고).
- `AgentView.health/hunger` 로 머리 위 상태 바 렌더링.
- `lastAction` 별 애니메이션(gather=캐기, trade=악수) 매핑.
- LLM 에이전트로 실행하려면 서버의 policy_factory 만 `LLMPolicy` 로 교체 —
  Unity 측 변경은 없다(같은 스냅샷 계약).
