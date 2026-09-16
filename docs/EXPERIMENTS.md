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
