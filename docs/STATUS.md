# 현재 목표

최종 갱신: 2026-09-17

궁극적으로 상위 memory가 `sub_task`를 제공하고, low-level agent가 현재 RGB와
현재 보이는 semantic metadata로 해당 subtask의 action chunk를 실행한다.

현재는 memory를 구현하거나 평가하지 않는다. `run_agent_loop(controller,
sub_task, max_steps=10)` 바닐라 기준선을 견고하게 만드는 단계다.

루프 내부에는 장기 memory와 구분되는 bounded RGB observation buffer만 둔다.
`image_history_steps`로 길이를 정하고 최신 프레임부터 모델에 전달한다.

완료 조건은 다음과 같다.

- 매 tick 하나의 허용된 action만 실행한다.
- 현재 보이는 임시 object reference만 조작에 사용할 수 있다.
- 완료 토큰이면 즉시 종료한다.
- action 예산 소진 시 마지막 관측으로 성공 여부를 반환한다.
- 잘못된 JSON, 금지 action, simulator 오류를 안전하게 종료 처리한다.

현재 단위 테스트 18개가 통과한다. 실제 단일 진단에서 `Turn right once`는
1 action 후 완료했고, `Pick up the apple`은 회전→집기→완료로 성공했다. 이는
각 1회 진단이며 성공률 benchmark는 아니다.

직접 subtask를 입력하는 최소 웹 콘솔을 제공한다. 1인칭 정책 관측과 UI 전용
3인칭 추적 카메라를 동시에 표시하며, 카메라 제어 action은 정책 feedback에서
격리된다.
