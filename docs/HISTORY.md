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
