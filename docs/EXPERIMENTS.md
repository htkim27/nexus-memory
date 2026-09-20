# 실험 기록

모두 2026-09-17, `FloorPlan1`, 실행 중인 Gemma 4 12B NF4 서버를 사용한 단일
개발 진단이다. 반복 성공률 benchmark가 아니다. 원본은 `runs/`의 local-only
JSON report다.

- `Turn right once`: 최초에는 잘못된 `TurnRight` 출력으로 실패했고, action 이름
  계약 보완 뒤에는 2회 회전하는 과잉 행동을 발견했다. 완료 규칙과 RGB buffer를
  적용한 최종 실행은 1회 `RotateRight` 후 완료했다.
  [최종 report](../runs/vanilla/20260917-turn-right-image-history/report.json),
  SHA-256 `ac6a2c66...e45e46`.
- `Pick up the apple`: 초기에는 Egg를 집고 허위 완료했다. 대상 일치 guard와
  안전한 schema 정규화 적용 후 최종 실행은 `RotateLeft` → `PickupObject(Apple)`
  → `complete`로 성공했다.
  [최종 report](../runs/vanilla/20260917-pickup-apple-parser-fix/report.json),
  SHA-256 `e98915bc...6054e`.

중간 실패 report도 각각의 `runs/vanilla/20260917-*` 디렉터리에 보존되어 있다.

- UI/observer smoke: `Turn right once`에서 초기·실행·최종 화면까지 3회 UI
  update가 발생했고 `RotateRight` 1회 후 성공했다. 1인칭은 RGB 640×480,
  observer는 RGBA 640×480이었다. 단일 통합 확인이며 별도 영구 artifact는
  저장하지 않았다.

## 2026-09-20 · ALFRED FloorPlan15 원본 규칙 프로토타입

공통 환경: ALFRED 코드 `f91f4c0c96c7a29f33d0557f86b0a21035379b3b`,
AI2-THOR 2.1.0, Python 3.7.12, 공식 `json_2.1.0` 아카이브 SHA-256
`c2e9e4256f94ccb2451dcaf99ff26124777eda0f256110489d1f0eaf6a3f02e9`,
`DISPLAY=:0`, FloorPlan15/train, annotation 0. 과제별 JSON 해시와 원본
계획은 [manifest](../alfred/tasks.json)에 있다. 모델 실행은 모두
`google/gemma-4-12B-it`, revision
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, NF4 4-bit,
`max_new_tokens=512`, `do_sample=false`였고 응답별 token 수는 각
`policy_response_*.json`에 있다. 모델 호출 제한은 두지 않았다.
원본/대조 실행에는 모델을 쓰지 않았다. 정확한 과거 runner 소스 해시는
실행 중 미커밋 코드가 바뀌어 **unknown**이다. 현재 코드부터는 실행 요약에
runner·manifest 해시를 기록한다. 행동·실패는 모두 episode 누적치이며,
시뮬레이터 내부 추가 호출은 별도로 이벤트에 기록했다.

| 실행 | 정책 행동/실패/모델 호출 | 예산(행동/실패) | 공식 판정과 근거 |
|---|---:|---:|---|
| [전문가 와인병](../runs/alfred/expert-wine-1/summary.json) | 26/0/0 | 1000/10 | 성공, 1/1 |
| [전문가 머그잔](../runs/alfred/expert-mug-1/summary.json) | 28/0/0 | 1000/10 | 성공, 3/3; 세척 이력 4개 |
| [전문가 사과](../runs/alfred/expert-apple-1/summary.json) | 109/0/0 | 1000/10 | 성공, 4/4; 가열 이력 3개 |
| [와인병 마지막 배치 생략](../runs/alfred/control-wine-no-put-1/summary.json) | 25/0/0 | 1000/10 | 실패, 0/1 |
| [머그잔 마지막 배치 생략](../runs/alfred/control-mug-no-put-1/summary.json) | 27/0/0 | 1000/10 | 실패, 1/3; `Pass`·`CleanObject` 내부 실행 확인 |
| [사과 마지막 배치 생략](../runs/alfred/control-apple-no-put-1/summary.json) | 108/0/0 | 1000/10 | 실패, 3/4; 가열 이력만으로 최종 성공 아님 |
| [칼 없이 절단](../runs/alfred/control-apple-no-knife-1/summary.json) | 31/1/0 | 1000/10 | 실패, 0/4; 31번 행동에서 공식 칼 소지 검사 거부 |
| [잘못된 컵에 배치](../runs/alfred/control-wine-wrong-receptacle-1/summary.json) | 26/1/0 | 1000/10 | 실패, 0/1; 공식 `PutObject` 거부 |
| [와인병 조기 완료](../runs/alfred/control-wine-early-complete-1/summary.json) | 0/0/0 | 1000/10 | 실패, 0/1 |
| [와인병 탐색 1](../runs/alfred/vlm-wine-explore-1/summary.json) | 16/0/16 | 60/10 | 중단; `LookDown` 반복, 최종 판정 unknown |
| [와인병 탐색 2](../runs/alfred/vlm-wine-explore-2/summary.json) | 9/3/12 | 60/10 | 형식 오류 `LookAround` 재출력, 최종 판정 unknown |
| [와인병 탐색 3](../runs/alfred/vlm-wine-explore-3/summary.json) | 60/0/61 | 60/10 | 실패, 0/1; [수정 계획 v1](../alfred/plans/wine_v1.json) |
| [머그잔 탐색 1](../runs/alfred/vlm-mug-explore-1/summary.json) | 40/0/40 | 40/10 | 실패, 1/3; 회전 반복 |
| [머그잔 탐색 2](../runs/alfred/vlm-mug-explore-2/summary.json) | 14/10/14 | 60/10 | 실패, 1/3; [계획 v1](../alfred/plans/mug_v1.json), 충돌 반복 |
| [머그잔 탐색 3](../runs/alfred/vlm-mug-explore-3/summary.json) | 16/0/16 | 100/10 | 실패, 1/3; [계획 v2](../alfred/plans/mug_v2.json), 회전 16회에서 진단 종료 |
| [사과 탐색 1](../runs/alfred/vlm-apple-explore-1/summary.json) | 40/0/40 | 40/10 | 실패, 0/4; 회전 반복 |
| [사과 탐색 2](../runs/alfred/vlm-apple-explore-2/summary.json) | 60/0/60 | 60/10 | 실패, 0/4; [계획 v1](../alfred/plans/apple_v1.json), 회전 반복 |
| [사과 탐색 3](../runs/alfred/vlm-apple-explore-3/summary.json) | 16/0/16 | 200/10 | 실패, 0/4; [계획 v2](../alfred/plans/apple_v2.json), 회전 16회에서 진단 종료 |

모든 링크의 원시 JSONL·RGB·mask·HTTP 요청/응답은 `runs/alfred/`의
**local-only artifact**다. 각 파일의 SHA-256, 모델/계획 정보와
요청 감사 결과는 추적되는 [results.json](../alfred/results.json)에 있다.
와인병 탐색 1~3은 초기 구현이 현재 RGB와 과거 RGB **최대 5장**을 전달해
요구된 과거 4장 제한을 위반했다. 이를 발견하고 현 코드에서 수정했다.
머그잔·사과 탐색의 실제 요청은 총 최대 5장, 300×300 PNG와 prompt만
담긴 것으로 감사됐다. 첫 두 와인병 실행은 최종 공식 평가가 이뤄지지
않았으므로 성공/실패 라벨을 추정하지 않는다. 탐색은 과제당 3회 상한에
도달했고, VLM 성공 후보나 고정 계획의 3회 재검증은 없다. 단위 테스트와
이 단일 씬 진단을 ALFRED benchmark나 MEM 재현 성능으로 해석하지 않는다.

## 2026-09-20 · VLM 로그 재분석 및 계측 변경 (episode 없음)

`runs/alfred/vlm-*-explore-*/events.jsonl`과 실제 `policy_request_*`/
`policy_response_*` local-only artifact를 읽은 사후 분석이다. 새 simulator나
VLM episode는 실행하지 않았고, 추가 탐색 예산도 소비하지 않았다. 9개 탐색의
271 정책 행동은 회전/시선 230회, 전진 41회, bbox 조작 0회였다. 실패한 전진
13회는 와인병 `Cube.527` 3회와 머그잔 `Pan_7e5e2cad` 10회의 반복이었다.
머그잔·사과의 요청은 image 1–5장, 각 300×300 PNG와 prompt만 담긴 것으로
재확인했다. 와인병 탐색 1–3의 image 6장 위반은 기존 표의 실패/중단 증거로
그대로 남긴다.

현 [runner](../alfred/runner.py)는 다음 실행의 action event에 RGB 변화율을
기록하고, 프롬프트에 이미지 순서·현재 RGB bbox grounding·scan 반복 금지를
명시한다. 코드 SHA는 아직 새 episode artifact가 없어 적용 대상에 기록할 수
없다. 검증은 `uv run pytest -q` 29개와 `uv run ruff check .` 통과이며,
이는 단위/정적 검증일 뿐 rollout 또는 benchmark 증거가 아니다.

## 2026-09-20 · 승인된 추가 탐색 진단

공통 환경은 위 ALFRED FloorPlan15 설정과 동일하다. 모델은
`google/gemma-4-12B-it`, revision
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, NF4 4-bit,
`max_new_tokens=512`, `do_sample=false`이다. 사용자 승인 후 각 과제 1회씩
원본 초기 상태에서 실행했다. 예산은 정책 행동 100·실행 실패 10이며,
`--repeat-action-limit 4`로 같은 행동 4회 반복 시 조기 진단 종료했다.
이는 고정 계획의 1,000/10 공식 재검증이 아니다.

| 실행 | 정책 행동/실패/모델 호출 | 공식 판정과 근거 | 요청 감사 |
|---|---:|---|---|
| [와인병 진단](../runs/alfred/vlm-wine-diagnostic-20260920-1/summary.json) | 4/3/4 | 실패, 0/1; `MoveAhead` 4회, `Cube.527` 충돌 3회 | 4회, 최대 4 image, 통과 |
| [머그잔 진단](../runs/alfred/vlm-mug-diagnostic-20260920-1/summary.json) | 5/3/5 | 실패, 1/3; `RotateLeft` 뒤 `MoveAhead` 4회, `Pan_7e5e2cad` 충돌 3회 | 5회, 최대 5 image, 통과 |
| [사과 진단](../runs/alfred/vlm-apple-diagnostic-20260920-1/summary.json) | 4/0/4 | 실패, 0/4; RGB가 변했는데도 `RotateLeft` 4회 | 4회, 최대 4 image, 통과 |

새 runner SHA-256은 `49aa2b9a2e61e1be049a1bfcd15c38397bc63f106fb9c523b673fd4e866ae93f`,
manifest SHA-256은 `61c2addbec863a2b7b864bceef52f7aff008b6391974edcb7a5b24c022496084`다.
세 raw artifact는 `runs/alfred/`의 Git 무시 local-only 자료다. VLM 성공과
고정 계획 재검증은 여전히 0회이고, 이후 탐색은 새 예산을 합의한 뒤에만 한다.
