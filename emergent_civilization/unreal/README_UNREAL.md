# Unreal Engine 클라이언트 — 작은 마을 시각화 (C++)

Python 시뮬레이션(`server/viz_server.py`)이 스트리밍하는 월드 스냅샷을
Unreal에서 3D로 렌더링한다. **Unreal은 순수 프레젠테이션 계층이며 시뮬레이션의
결정에 관여하지 않는다** — 브라우저 뷰어·Unity 클라이언트와 정확히 같은
`GET /state` 엔드포인트를 소비한다.

> ⚠️ **정직하게 말씀드립니다**: 이 코드는 실제 Unreal 에디터에서 컴파일 검증을
> 하지 못한 상태로 작성됐습니다(이 환경엔 Unreal이 없음). API는 최대한 정확히
> 썼지만, Unity 스크립트 때보다 컴파일 오류가 날 가능성이 더 높습니다. 에러
> 메시지를 그대로 보내주시면 바로 고쳐드리겠습니다.

## 이번 버전의 범위 (v1)

**포함**: 서버 연결·JSON 파싱·기본 도형(원기둥/구/원뿔/큐브)으로 마을 구성·
낮/밤 조명 변화·에이전트 이동 보간·이름표.

**의도적으로 뺀 것**: 에이전트별 색상 구분(머티리얼 파라미터가 프로젝트마다
달라 안전하게 자동화하기 어려움), 커스텀 궤도 카메라(Unreal 프로젝트마다
입력 설정이 달라 에디터 기본 플라이 카메라를 쓰는 게 더 안전함). 파이프라인이
확실히 붙은 다음, 실제 아트 에셋(Quixel Megascans 등)으로 교체하면서 같이
다듬어나가는 걸 추천합니다.

## 준비물

- **Unreal Engine 5.x**, C++ 프로젝트로 생성(Blueprint-only 아님).
- **Visual Studio 2022** + "게임 개발(C++)" 워크로드 (Windows 컴파일용).
- **실행 중인 Python 서버**:
  ```bash
  cd emergent_civilization
  python -m server.viz_server        # http://localhost:8000
  ```

## 설정

### 1. C++ 프로젝트 생성
Unreal 프로젝트 브라우저 → **Games → Blank** (또는 원하는 템플릿) →
**C++** 선택(Blueprint 아님) → 프로젝트 생성.

### 2. 소스 파일 복사
`unreal/Source/` 의 파일 전부(`.h`/`.cpp` 12개)를 프로젝트의
`Source/<프로젝트이름>/` 폴더로 복사.

### 3. 모듈 의존성 추가
프로젝트의 `Source/<프로젝트이름>/<프로젝트이름>.Build.cs` 파일을 열어
`PublicDependencyModuleNames.AddRange` 안에 다음 세 개를 추가:
```csharp
"HTTP", "Json", "JsonUtilities"
```
(기존에 `"Core", "CoreUObject", "Engine", "InputCore"` 가 이미 있을 텐데,
그건 그대로 두고 세 개만 추가하면 된다.)

### 4. 컴파일
Visual Studio에서 프로젝트 열고(또는 `.uproject` 우클릭 →
**Generate Visual Studio project files**) 빌드. 또는 Unreal 에디터에서
**Compile** 버튼.

### 5. 레벨에 액터 배치
1. Content Browser → **C++ 클래스** 폴더 → `SimClient`, `VillageManager` 를
   레벨 뷰포트로 드래그.
2. 레벨에 **Directional Light** 가 없다면 하나 추가(대부분 기본 템플릿엔 있음).
3. 배치한 `VillageManager` 선택 → Details 패널에서 **Sun** 슬롯에 그
   Directional Light 연결.
4. 배치한 `SimClient` 선택 → Details 패널에서:
   - **Server Url** = `http://localhost:8000`
   - **World Size**(따로 없음 — 스냅샷의 width/height로 자동 처리됨)
   - **Village Manager** 슬롯에 2에서 배치한 `VillageManager` 연결.
5. ▶ **Play**. 에디터 뷰포트 기본 조작(우클릭+WASD로 날아다니기)으로 둘러본다.

## 좌표 규약
- 월드 1칸 = Unreal 100 유닛(cm). `(grid_x*100, grid_y*100, 0)` 에 배치.
- 에이전트는 매 프레임 `VInterpTo` 로 목표 위치까지 부드럽게 이동.
- 노드 프롭은 잔량(`amount/capacity`)에 비례해 크기가 변한다.

## 스냅샷 스키마
`unity/README_UNITY.md` 의 스키마와 동일 (`sim/snapshot.py` 가 유일한 출처).
`FJsonObjectConverter::JsonObjectStringToUStruct` 가 `SnapshotModels.h` 의
USTRUCT 들로 자동 매핑한다.

## 문제 해결
- 컴파일 에러 → 정확한 에러 메시지(파일명·줄 번호 포함) 그대로 보여주면 고쳐드림.
- 아무것도 안 보임 → 서버가 켜져 있는지, Output Log에 `[SimClient]` 경고가
  있는지 확인.
- 다음 단계(실제 아트 에셋, 에이전트 색상 구분, 커스텀 카메라)는 파이프라인
  확인되면 이어서 작업.
