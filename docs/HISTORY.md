# 변경 기록

## 2026-09-17 · 바닐라 low-level 기준선으로 초기화

기존 memory/high-level/고정 skill 구조를 현재 실행 범위에서 제거했다. 이전
구조가 이후 low-level 설계에 선입견을 주지 않도록 문서도 현재 목표만 남겼다.
새 기준선은 하나의 `sub_task`를 RGB와 제한된 visible metadata로 수행하는
`run_agent_loop`다.

## 2026-09-17 · bounded RGB 시계열과 실행 guard

`image_history_steps` 크기의 RGB buffer를 추가해 최신 프레임부터 모델에
전달한다. 좌표가 포함된 THOR object ID는 임시 reference로 가렸고, 다른 종류의
객체 pickup과 손이 찬 상태의 pickup을 실행 전에 거부한다. 실제 진단에서 발견한
action 이름·완료 반복·출력 schema 문제를 prompt와 parser에 반영했다.

## 2026-09-17 · 직접 입력 콘솔과 3인칭 observer

`run_agent_loop`를 감싸는 최소 Gradio UI를 추가했다. 사용자가 subtask와 action
예산, RGB history 길이를 직접 입력한다. AI2-THOR third-party camera는 에이전트
뒤·위에서 갱신하지만, camera update event가 low-level 정책의 실제 action 결과를
덮어쓰지 않도록 controller proxy에서 분리한다.

## 2026-09-20 · ALFRED 원본 단일 씬 오라클 유도 환경과 하네스

THOR 5.0 자체 subtask 규칙과 분리해 ALFRED 원본 커밋
`f91f4c0c96c7a29f33d0557f86b0a21035379b3b`, Python 3.7,
AI2-THOR 2.1.0 환경을 구성했다. 공식 `json_2.1.0` 아카이브와 고정된
FloorPlan15/train 세 trial의 출처·해시를 manifest에 남겼다. 구버전
PyTorch 1.1.0을 포함한 공식 requirements가 설치됐고, 원본 Unity build는
기존 `DISPLAY=:0`에서 실행됐다. 의미·행동·평가기 코드는 수정하지 않았다.

ALFRED 전용 runner는 공식 `reset → restore_scene → init_action → set_task`,
`va_interact`, `get_transition_reward`, `get_goal_satisfied`,
`get_goal_conditions_met`을 호출한다. RGB와 언어 지시를 HTTP VLM에 전달하고,
정수 bbox를 300×300 mask로 바꾼다. 정책과 격리된 상태·객체 ID·내부
시뮬레이터 호출을 JSONL에 보존한다. 완료 선언은 공식 목표/서브골 검증
요청으로 다루며, 성공 라벨로 직접 사용하지 않는다.

전문가 재생 세 건은 각각 26·28·109행동, 실행 실패 0회, 공식 목표 성공이었다.
최종 배치 생략 세 건, 칼 생략 절단, 잘못된 컵에 배치 시도, 조기 완료는
공식 목표 실패였다. 수도 조작에는 `Pass`와 `CleanObject`가 내부 호출됐고,
절단 후 세 조각이 생성되어 하나가 후속 pickup 대상이 된 것을 확인했다.
이는 공식 평가기에 대한 단일 씬 대조이며 전체 benchmark 증거는 아니다.

Gemma 4 12B NF4 서버의 `/health`와 `/generate`를 연결했다. 각 과제에
원본 지시와 수정 계획을 사용한 탐색 3회를 실행했으나 VLM 공식 성공은 0회다.
회전·전진 충돌 반복, 허용되지 않은 `LookAround` 출력이 관찰됐다.
와인병 탐색 세 건은 초기 이미지 history 구현이 현재 RGB 외 과거 5장을
보낸 사실을 요청 감사에서 발견했다. 현 구현은 과거 4장으로 고쳤으나
기존 탐색을 계약 준수 사례로 재분류하지 않는다. 해당 runs는 실패·중단
자료로 유지한다. 실행 중 코드가 바뀌어 과거 runner 파일 해시는 알 수 없다.
향후 요약에는 runner와 manifest 해시를 자동으로 기록한다.

`uv run pytest -q` 27개와 `uv run ruff check .`가 통과했다. 원본 실행
설치·재생의 한계는 Python 3.7/Unity build의 구버전 의존성과 단일
FloorPlan15 검증이다. VLM 고정 계획 3회 재검증과 오라클 등록은 미완료다.
실행별 예산·모델 설정·local-only artifact는
[EXPERIMENTS.md](EXPERIMENTS.md)와 [결과 요약](../alfred/results.json)에 있다.

## 2026-09-20 · VLM 요청/행동 재분석과 다음 탐색 전 계측 보완

추가 VLM 실행 없이 `runs/alfred/vlm-*-explore-*`의 실제 HTTP 요청·응답과
`policy_action` 이벤트를 재분석했다. 9개 탐색의 271 정책 행동은 회전/시선
230회와 전진 41회였고, bbox 조작 출력은 없었다. 와인병은 `Cube.527`에 대한
전진 실패 3회, 머그잔은 `Pan_7e5e2cad`에 대한 전진 실패 10회를 반복했다.
따라서 현재 자료는 대상 mask/bbox의 실패가 아니라 탐색을 끝내고 대상으로
접근하기 전의 정책 루프를 보여 준다. 머그잔·사과 요청은 총 최대 5장의
300×300 PNG와 prompt만 포함했고, 와인병의 과거 5장 버그는 기존 기록대로
계약 위반 자료로 유지한다.

다음 실행에서 현재 프레임과 과거 프레임의 역할, 현재 RGB에서만 bbox를
선택해야 한다는 조건, 완성된 시각 scan 뒤의 반복 금지를 프롬프트에 더
명시했다. 모든 정책/전문가 action event에 RGB 변화율도 남기도록 했다.
이는 객체 metadata·정답 mask·평가기 상태를 정책에 추가하지 않으며, 행동을
전문가 primitive로 대체하지 않는다. `uv run pytest -q` 29개와
`uv run ruff check .`가 통과했다. 서버는 중지 상태로 유지했고, 탐색 3회
상한을 소진한 상태이므로 사용자와 새 예산 및 검증 계획을 합의하기 전에는
episode를 실행하지 않았다.

## 2026-09-20 · 승인된 추가 VLM 진단 3건

사용자가 새 진단을 진행하도록 승인한 뒤, 각 과제에 1회씩 원본 초기 상태에서
실행했다. 공통 예산은 100 정책 행동·10 실행 실패이고, 같은 정책 행동이 4회
연속되면 진단 종료했다. 모델은 `google/gemma-4-12B-it`, revision
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, NF4 4-bit,
`max_new_tokens=512`, `do_sample=false`였으며 서버는 종료했다.

와인병은 `MoveAhead` 4회 중 3회가 `Cube.527` 충돌로 실패했고, 머그잔은
`RotateLeft` 뒤 `MoveAhead` 4회 중 3회가 `Pan_7e5e2cad` 충돌로 실패했다.
사과는 RGB 변화율 0.969–0.991인 서로 다른 화면에서도 `RotateLeft` 4회를
선택했다. 세 실행 모두 공식 목표 실패, bbox 조작 0회, 성공 후보 0회다.
요청 감사는 와인병 4회/최대 image 4장, 머그잔 5회/5장, 사과 4회/4장으로
통과했다. 이는 새 프롬프트와 계측의 단일 진단 근거이지 benchmark나 재검증
근거가 아니다. 새 결과는 [results.json](../alfred/results.json)에, JSONL·RGB·
HTTP artifact는 Git 무시 local-only `runs/alfred/`에 있다. 추가 episode는
새 예산 합의 전에는 실행하지 않는다.
