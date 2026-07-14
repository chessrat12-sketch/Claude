# Unity 3D 클라이언트 — 작은 마을 시각화

Python 시뮬레이션(`server/viz_server.py`)이 스트리밍하는 월드 스냅샷을
Unity에서 3D로 렌더링한다. **Unity는 순수 프레젠테이션 계층이며 시뮬레이션의
결정에 관여하지 않는다** — 브라우저 뷰어(`viewer/village.html`)와 정확히 같은
`GET /state` 엔드포인트를 소비한다.

## 준비물
- **Unity 2021 LTS 이상** (없으면 [Unity Hub](https://unity.com/download) 설치 →
  Hub에서 2021/2022 LTS 에디터 설치). JsonUtility·UnityWebRequest는 기본 내장,
  외부 패키지 불필요.
- **실행 중인 Python 서버** (Unity를 켜기 전에 먼저 실행):
  ```bash
  cd emergent_civilization
  python -m server.viz_server        # http://localhost:8000
  ```

## 설정 — 한 번의 컴포넌트 추가 (약 2분)

**GameObject 를 손으로 배치·연결할 필요가 없다.** `SceneBootstrap` 이
카메라·조명·지면·렌더러·클라이언트를 실행 시점에 전부 만들고 연결한다.

1. Unity Hub에서 **새 3D 프로젝트** 생성 (Built-in/URP 무관).
2. `unity/Scripts/` 폴더를 통째로 프로젝트 `Assets/` 로 복사
   (5개 파일: `SceneBootstrap`, `CameraOrbit`, `SimClient`, `VillageRenderer`,
   `SnapshotModels`).
3. 빈 씬에서 **GameObject → Create Empty** → 그 오브젝트에
   **Add Component → "Scene Bootstrap"**.
   - `Server Url` = `http://localhost:8000` (기본값)
   - `World Size` = 서버 `--size` 와 동일 (기본 14)
   - `Poll Interval` = 서버 `--tick-ms` 와 맞춤 (기본 0.3)
4. ▶ **Play**. 마우스 **좌드래그=회전, 휠=줌, 우드래그=이동**으로 3D 마을을 둘러본다.

프리팹을 지정하지 않으면 렌더러가 **프리미티브**(캡슐=에이전트/포식자,
실린더=나무, 구=돌, 큐브=은신처, 플레인=지면)로 즉시 마을을 구성한다. 낮/밤에
따라 조명이 바뀌고, 에이전트 머리 위에 체력바가, 사건(거래·선물·건축·물림)이
떠오르는 라벨로 표시된다. 아트를 붙이려면 `VillageRenderer` 의 프리팹 슬롯
(`Tree/Rock/Agent/Shelter/Predator Prefab`)에 프리팹을 넣으면 된다.

> **수동 설정(고급)**: 원한다면 `Village`(VillageRenderer)·`SimClient` 를 직접
> 만들어 연결해도 된다 — `SceneBootstrap` 이 하는 일을 손으로 하는 것뿐이다.

## 연결이 안 될 때 (체크리스트)
- Console에 `SimClient` 오류 → **서버가 실행 중인지** 먼저 확인
  (`python -m server.viz_server`, 브라우저로 http://localhost:8000 열어보기).
- 아무것도 안 보임 → `World Size` 가 서버 `--size` 와 같은지, Scene 뷰가 아니라
  **Game 뷰**를 보고 있는지 확인.
- 원격 서버라면 `Server Url` 을 해당 호스트로. (localhost 는 같은 PC 기준.)

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
