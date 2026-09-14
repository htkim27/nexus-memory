# Nexus Memory

AI2-THOR와 Vision-Language Model(VLM)을 이용해 **Embodied Agent의 장기 기억 구조가 장기 과제의 성공률·행동 효율·추론 비용에 미치는 영향**을 비교하는 연구 프레임워크입니다.

이 프로젝트는 MEM(Multi-Scale Embodied Memory)의 고수준 기억 루프에서 출발하지만, 실제 로봇의 Low-Level VLA 연속 제어는 AI2-THOR의 이산 행동 API로 대체합니다. 연구의 초점은 제어기 자체가 아니라 다음 질문에 있습니다.

> 동일한 VLM, 관찰, 행동 공간, 평가 조건과 예산에서 기억 구조만 바꾸었을 때 장기 과제 수행 능력이 달라지는가?

현재 저장소에는 RGB-only 환경 경계, 행동 변환기, 물리 실패 피드백을 보존하는 두 기억 구조, VLM 어댑터, 비공개 상태 평가기와 Gradio 대시보드의 기반 구현이 포함되어 있습니다. 아직 벤치마크 결과나 MEM 원 논문의 재현 결과는 제공하지 않습니다.

## 새로운 방향

장기 기억 구조를 공정하게 비교하려면 기억 외의 실패 요인을 먼저 통제해야 합니다. Nexus Memory는 다음 세 문제를 프레임워크의 독립 모듈로 취급합니다.

1. **Action Grounding**: VLM의 자연어 하위 과제를 실행 가능한 AI2-THOR 명령으로 제한합니다.
2. **Physical Preconditions**: 도구 소지, 거리, 가시성, 용기 상태 등의 조건 때문에 행동이 실패하면 실제 오류를 다음 의사결정에 반영합니다.
3. **Leak-free Evaluation**: 성공 판정에는 전체 simulator metadata를 사용하지만, 해당 정보는 VLM 입력에서 격리합니다.

이 구조를 통해 “기억이 나빠서 실패한 경우”와 “행동 변환·물리 실행·평가 설계가 잘못되어 실패한 경우”를 구분할 수 있는 실험 기반을 만드는 것이 목표입니다.

## 시스템 구조

```text
Global Goal ─────────────────────────────────────────────┐
                                                        ▼
RGB frame + Memory Context ──► VLM Policy ──► Text Subtask
                                      │                 │
                                      │ memory rewrite  ▼
                                      │          Action Translator
                                      │                 │ strict THOR JSON
                                      │                 ▼
                                      └────── Memory ◄── AI2-THOR
                                                ▲       │
                              success/error ─────┘       │ hidden metadata
                                                        ▼
                                                  Task Evaluator
```

한 step의 실행 순서는 다음과 같습니다.

1. VLM은 현재 RGB, 전역 목표, 선택한 메모리의 문자열 context만 받습니다.
2. VLM은 갱신된 메모리 초안과 자연어 하위 과제 하나를 JSON으로 생성합니다.
3. `ActionTranslator`가 하위 과제와 현재 보이는 object ID 목록을 AI2-THOR 명령으로 변환합니다.
4. AI2-THOR가 명령을 실행하고 `lastActionSuccess`, `errorMessage`를 반환합니다.
5. 선택한 메모리 구조가 VLM의 초안과 실제 실행 결과를 함께 기록합니다.
6. `TaskEvaluator`가 VLM에 공개되지 않은 metadata로 목표 달성 여부와 completion score를 계산합니다.

## 정보 접근 경계

| 정보 | VLM | Translator | Memory | Evaluator |
| --- | :---: | :---: | :---: | :---: |
| 현재 RGB 프레임 | O | X | 간접 요약만 | X |
| 전역 목표 | O | 하위 과제만 | O | O |
| 현재 보이는 object ID | X | O | 실행 기록만 | O |
| 전체 object metadata | X | X | X | O |
| `lastActionSuccess`, `errorMessage` | 다음 step의 memory를 통해 O | X | O | X |

`ThorEnvironment.observe()`와 공개 step 결과에는 RGB와 실행 feedback만 포함됩니다. 전체 metadata 접근은 grounding과 평가를 위한 backend 전용 경로로 분리되어 있습니다. 디버깅 과정에서도 metadata를 VLM prompt에 넣으면 기본 실험 조건이 아닌 oracle 조건으로 취급해야 합니다.

## 모듈

| 파일 | 역할 | 현재 상태 |
| --- | --- | --- |
| `environment.py` | AI2-THOR 생명주기, RGB 관찰, 실행 feedback, 비공개 metadata 경계 | 구현됨 |
| `translator.py` | 자연어 하위 과제를 검증된 AI2-THOR action dictionary로 grounding | 규칙 기반 구현, 경량 LLM mapper 주입 가능 |
| `memory.py` | 공통 memory interface와 MEM-Flat/Hierarchical 구현 | 기반 구현됨 |
| `agent.py` | VLM backend, 구조화 출력 파싱, token/latency 계측 | Gemma 3 및 demo backend 구현됨 |
| `evaluator.py` | metadata predicate 기반 성공 판정과 부분 점수 | 기반 predicate/registry 구현됨 |
| `app.py` | 전체 loop를 조작하고 관찰하는 Gradio 대시보드 | 구현됨 |

### Action Translator

규칙 기반 translator는 다음 범주의 명령을 지원합니다.

- 이동·시점: `MoveAhead`, `MoveBack`, `RotateLeft`, `RotateRight`, `LookUp`, `LookDown`, `Pass`
- 물체 조작: `PickupObject`, `PutObject`, `OpenObject`, `CloseObject`
- 상태 변경: `ToggleObjectOn/Off`, `SliceObject`, `BreakObject`, `CleanObject`, `DirtyObject`, `FillObjectWithLiquid`, `EmptyLiquidFromObject`

물체 행동은 translator에 전달된 object ID와 정확히 일치해야 합니다. 허용되지 않은 action, 추가 argument, 보이지 않는 object ID는 실행 전에 거부됩니다. 규칙으로 해석할 수 없는 하위 과제도 임의의 행동으로 추측하지 않고 grounding failure로 기록합니다.

Gemma 3 prompt에는 translator가 지원하는 자연어 action 형식을 명시합니다. `Fridge`처럼 동사가 없는 대상명은 허용하지 않고 `Open the fridge`처럼 한 번에 실행 가능한 문장을 요구합니다. `subtask`는 구조화된 action object가 아니라 자연어 문자열로 유지합니다.

향후 경량 LLM translator를 사용할 때도 입력은 하위 과제와 object ID 목록으로 제한하고, 출력은 동일한 allow-list validator를 통과시켜야 합니다.

### Memory Architectures

#### MEM-Flat

`MemFlatMemory`는 이전 언어 기억을 매 step 재작성하는 autoregressive flat-text baseline입니다. VLM의 메모리 초안 뒤에 실제 실행 결과를 강제로 추가하므로, 동일 step에서 발생한 실패를 모델이 누락할 수 없습니다.

#### Hierarchical

`HierarchicalMemory`는 다음 tier의 경계를 제공하는 NexusSum-inspired 기반 구현입니다.

- 전역 목표/요약
- roll-up된 episode event
- 최근 working event

현재 roll-up은 결정론적 placeholder이며 학습된 계층 요약기를 구현한 것이 아닙니다. 추후 summarizer를 추가할 때 실패 원인, 미완료 조건, 물체의 마지막 위치와 상태를 보존하는 정책을 별도로 검증해야 합니다.

두 구조 모두 실패 시 simulator의 원문 `errorMessage`와 “전제 조건을 바꾸기 전 동일 행동을 반복하지 말 것”이라는 기록을 다음 VLM step에 제공합니다.

실행용 원본 action과 UI에는 정확한 `objectId`가 유지되지만, VLM에 전달되는 memory에는 `targetType`만 기록됩니다. 오류 문자열에 포함된 AI2-THOR object ID도 제거되므로 좌표가 memory를 통해 모델로 역류하지 않습니다.

### Task Evaluator

`TaskEvaluator`는 `isDirty`, `isSliced`, `isOpen`, `isToggled`, `isBroken`, `isFilledWithLiquid`, `isPickedUp`, `parentReceptacles` 같은 비공개 상태를 사용합니다. 전체 성공 여부와 만족한 조건의 비율인 `completion_score`를 함께 반환합니다.

자유 형식 목표 parser는 개발과 UI 확인을 위한 제한적 편의 기능입니다. 정식 실험에서는 목표 문자열에 의존하지 말고, 각 benchmark task에 사전 등록한 `GoalCondition` 목록을 사용해야 합니다. 해석하지 못한 goal clause는 성공으로 무시하지 않고 미충족 조건으로 계산합니다.

## 설치

이 저장소는 `uv`와 Python 3.12를 사용합니다.

```bash
uv sync --extra vlm --group dev
```

현재 lockfile은 다음 로컬 NVIDIA 환경에서 검증되었습니다.

| 구성 | 검증 환경 |
| --- | --- |
| OS | Ubuntu 26.04 LTS, x86_64 |
| GPU | NVIDIA GeForce RTX 5070 Ti 16 GB |
| Driver | 595.84 |
| Driver CUDA capability | 13.2 |
| PyTorch | 2.14.0+cu130 |
| Python | 3.12.14 |

RTX 50 시리즈의 Compute Capability 12.0을 지원하도록 `pyproject.toml`에서 PyTorch CUDA 13.0 index를 명시했습니다. PyTorch wheel은 CUDA runtime을 포함하므로 시스템에 설치된 CUDA toolkit과 정확히 같은 minor version일 필요는 없지만, 호환되는 NVIDIA driver는 필요합니다.

의존성의 기준 파일은 다음과 같습니다.

- `pyproject.toml`: 직접 의존성, VLM extra, 개발 도구, CUDA wheel source
- `uv.lock`: 재현 가능한 전체 dependency graph
- `requirements.txt`: `uv`를 사용하지 않는 환경을 위한 pip 호환 export
- `.python-version`: 프로젝트 Python 3.12 선택

lockfile을 의도적으로 갱신할 때는 다음 명령을 사용합니다.

```bash
uv lock --upgrade
uv export --extra vlm --no-dev --no-hashes --emit-index-url --output-file requirements.txt
```

## VLM 설정

기본 VLM은 멀티모달 instruction-tuned 모델인 `google/gemma-3-4b-it`입니다. PaliGemma 계열은 현재 기본 실험 대상에서 제외합니다.

Gemma 3 모델 파일을 받으려면 Hugging Face 모델 페이지에서 Google의 Gemma 사용 조건에 동의하고 인증해야 합니다.

```bash
uv run hf auth login
```

인증 후 별도 환경 변수 없이 실행하면 기본 모델이 로드됩니다.

```bash
uv run python app.py
```

`Gemma3VLMBackend`는 `AutoModelForImageTextToText`, Gemma 3 chat template와 BF16을 사용합니다. 여러 dashboard session이 같은 model instance를 공유하므로 episode를 다시 초기화해도 GPU에 모델을 중복 적재하지 않습니다.

모델을 받지 않고 UI와 simulator 연결만 확인하려면 demo backend를 명시합니다. Demo backend는 관찰을 이해하지 않고 회전 행동만 생성하므로 연구 결과에 사용하면 안 됩니다.

```bash
NEXUS_VLM_MODEL=demo uv run python app.py
```

## 대시보드 사용

```bash
uv run python app.py
```

대시보드에서 확인할 수 있는 항목은 다음과 같습니다.

- Global Goal과 AI2-THOR scene
- `MEM-Flat` 또는 `Hierarchical` memory 선택
- 실시간 RGB observation
- 현재 memory context와 VLM subtask
- 변환된 THOR action JSON
- 환경 실행 성공 여부와 원문 오류
- evaluator success/completion score
- 누적 token 수와 VLM latency

`Initialize`로 episode를 시작하고 `Step`으로 한 번 실행하거나 `Auto-Run`으로 최대 step 수까지 진행합니다. `Stop`은 현재 model/action step이 끝난 뒤 auto-run을 중단합니다.

각 browser/API session은 독립적인 AI2-THOR Controller를 사용합니다. 재초기화, evaluator 성공, auto-run 종료, session TTL 만료, 브라우저 session 삭제 또는 서버 종료 시 Controller를 닫습니다. 기본 idle TTL은 1시간이며 `NEXUS_SESSION_TTL_SECONDS`로 변경할 수 있습니다.

## 테스트와 검증

```bash
uv run pytest
uv run ruff format --check .
uv run ruff check .
```

현재 테스트는 다음 계약을 검사합니다.

- agent-facing 결과에 metadata가 포함되지 않는지
- object action이 제공된 ID에만 grounding되는지
- 잘못된 LLM mapper 출력이 실행 전에 차단되는지
- 두 memory 구조가 physical failure를 보존하는지
- 단일/복합 상태 목표와 receptacle 배치를 정확히 평가하는지

현재 검증 환경에서는 9개 테스트, CUDA GPU 연산, AI2-THOR `FloorPlan1` RGB 관찰 및 `RotateRight` 실행, Gradio UI build를 통과했습니다.

## 연구 프로토콜

### 비교 조건

MEM-Flat과 Hierarchical 비교 시 다음 항목을 동일하게 유지합니다.

- VLM checkpoint, precision, decoding parameter와 prompt
- scene, 초기 object 상태, seed와 목표 변경 event
- RGB 해상도와 관찰 이력 정책
- translator와 허용 action set
- evaluator predicate와 최대 environment step 수
- VLM에 전달되는 memory token budget

계층 요약에 추가 model call을 사용한다면 그 token과 latency도 전체 비용에 포함해야 합니다. 분석용 원본 로그, simulator metadata, 별도 지도나 object cache가 정책에 노출되면 해당 정보도 memory budget으로 간주합니다.

### 후보 과제군

| 과제 유형 | 장기 기억 부담 | 주요 평가 대상 |
| --- | --- | --- |
| 주방 청소 및 정리 | 여러 물체의 세척·수납 상태 | 누락과 중복 처리 |
| 저녁 식사 준비 | 재료 준비, 수량, 순서 | 단계 혼동과 완료율 |
| 장 본 물건 정리 | 물체별 적절한 보관 위치 | 탐색 및 위치 기억 |
| 식사 후 뒷정리 | 수거·세척·수납의 의존 관계 | physical precondition 처리 |
| 진행 중 인원 변경 | 기존 완료 상태를 보존한 계획 수정 | 목표 변경 반영 |

각 과제군은 한 번의 데모가 아니라 여러 scene/seed/초기 상태로 반복합니다. 현재 RGB 한 장만으로 전체 진행 상황을 복원할 수 없고, 과거의 탐색·행동·실패를 기억해야 성공하도록 구성해야 합니다.

### 주요 지표

- 전체 목표 성공률
- 조건 단위 completion score
- 환경 step 수와 중복 행동 수
- grounding failure와 physical action failure 수
- 실패 후 동일 행동 반복률
- VLM input/output token과 총 latency
- memory tier별 크기와 요약 호출 비용

성공률뿐 아니라 비용과 실패 유형을 함께 보고합니다. 두 방법 모두 쉽게 성공하거나 grounding/인식 실패가 대부분이면 해당 benchmark는 기억 구조를 비교하기에 적합하지 않은 것으로 판단합니다.

## 현재 한계

- 규칙 기반 translator는 제한된 영어 action phrase만 지원하며 범용 planner가 아닙니다.
- Hierarchical memory의 roll-up은 아직 학습형/LLM summarizer가 아닙니다.
- 자연어 goal evaluator는 제한된 template만 지원합니다.
- Gemma 3 backend는 아직 quantization과 KV-cache 크기를 자동 최적화하지 않습니다.
- 표준화된 task specification, episode initializer, batch runner와 결과 저장 형식은 아직 구현되지 않았습니다.
- 현재 결과를 실제 로봇의 연속 제어 성능이나 원래 MEM 시스템의 재현 결과로 해석할 수 없습니다.

## 로드맵

- [x] AI2-THOR RGB-only environment boundary와 action feedback
- [x] 규칙 기반 action grounding과 strict validator
- [x] physical failure를 보존하는 공통 memory interface
- [x] MEM-Flat 및 Hierarchical 기반 구현
- [x] metadata 기반 evaluator와 Gradio loop
- [x] CUDA 13.0 기반 `uv` 재현 환경
- [ ] 명시적 task specification 및 episode initializer
- [ ] 개발/평가 scene과 seed 분리
- [ ] batch experiment runner와 trajectory 저장 형식
- [ ] 공통 token budget 및 hierarchical summarizer 고정
- [ ] 후보 과제군별 predicate와 목표 변경 protocol 확정
- [ ] 여러 seed에 대한 paired A/B 평가
- [ ] 성공률·비용·실패 유형 통계와 ablation 보고

## 참고 자료

- [MEM: Multi-Scale Embodied Memory for Vision Language Action Models](https://arxiv.org/html/2603.03596v2)
- [NexusSum: Hierarchical LLM Agents for Long-Form Narrative Summarization](https://arxiv.org/html/2505.24575v1)
- [VLAs with Long and Short-Term Memory — Physical Intelligence](https://www.pi.website/research/memory)
- [AI2-THOR](https://github.com/allenai/ai2thor)
- [AI2-THOR Object State Changes](https://ai2thor.allenai.org/ithor/documentation/object-state-changes/)
- [ALFRED](https://github.com/askforalfred/alfred)
