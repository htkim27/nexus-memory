# ALFRED 원본 규칙 기반, 단일 씬의 Subtask 오라클 유도 프로토타입

현재 구현은 공식 ALFRED `f91f4c0c96c7a29f33d0557f86b0a21035379b3b`와
AI2-THOR 2.1.0을 별도 Python 3.7 프로세스에서 실행한다. `runner.py`는 원본
`ThorEnv`의 초기화, `va_interact`, 전이 보상, 최종 목표 판정을 그대로 호출한다.
기존 THOR 5.0 실행 경로에는 영향을 주지 않는다. 전문가 재생은 기준 자료이고,
VLM이 원본 초기 상태부터 공식 목표를 달성한 episode만 오라클 후보가 된다.

## 재현 환경

```bash
git clone https://github.com/askforalfred/alfred.git /home/htkim/dev/alfred-upstream
git -C /home/htkim/dev/alfred-upstream checkout f91f4c0c96c7a29f33d0557f86b0a21035379b3b
micromamba create -p /home/htkim/.local/share/mamba/envs/alfred-2.1 \
  --file alfred/conda-explicit.txt
/home/htkim/.local/share/mamba/envs/alfred-2.1/bin/python -m pip install \
  -r alfred/requirements.lock.txt
mkdir -p runs/alfred/source
curl -L --fail -o runs/alfred/source/json_2.1.0.7z \
  https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/json_2.1.0.7z
/home/htkim/.local/share/mamba/envs/alfred-2.1/bin/7zz x -y \
  -oruns/alfred/source runs/alfred/source/json_2.1.0.7z
python alfred/manifest.py \
  --data-root runs/alfred/source/json_2.1.0 \
  --split-file /home/htkim/dev/alfred-upstream/data/splits/oct21.json \
  --archive runs/alfred/source/json_2.1.0.7z \
  --output alfred/tasks.json
```

현재 머신의 원본 Unity 실행 파일은
`~/.ai2thor/releases/thor-201909061227-Linux64/thor-201909061227-Linux64`다.
SHA-256은 `329400f128d0c08126aa1409e931573b0e676e76395d2845e1338b2c38bf25ca`.
공식 아카이브 SHA-256은
`c2e9e4256f94ccb2451dcaf99ff26124777eda0f256110489d1f0eaf6a3f02e9`.
화면은 기존 `DISPLAY=:0`을 사용했다. 원본 `ThorEnv`의 300×300 RGB와
객체 segmentation 렌더링 설정을 유지했다. `conda-explicit.txt`는 conda
패키지의 URL과 해시를 고정하고, `requirements.lock.txt`는 실제 설치된 pip
버전을 기록한다.

## 실행

프로젝트 루트에서 실행한다. 결과 폴더는 새 이름이어야 한다.

```bash
ALFRED_ROOT=/home/htkim/dev/alfred-upstream \
  /home/htkim/.local/share/mamba/envs/alfred-2.1/bin/python \
  alfred/runner.py expert --task wine_bottle \
  --output runs/alfred/expert-wine-repeat

uv run python inference_2.py  # 별도 터미널의 기존 VLM 서버
ALFRED_ROOT=/home/htkim/dev/alfred-upstream \
  /home/htkim/.local/share/mamba/envs/alfred-2.1/bin/python \
  alfred/runner.py vlm --task wine_bottle \
  --plan-file alfred/plans/wine_v1.json \
  --output runs/alfred/vlm-wine-repeat
```

`--task`는 `wine_bottle`, `clean_mug`, `heated_apple_slice` 중 하나다.
`--max-policy-actions`와 `--max-failures`는 진단을 조기 종료할 때만 줄인다.
공식 상한 1,000행동·10회 실행 실패를 넘길 수 없다.
`--repeat-action-limit 16`은 한 행동을 계속 반복하는 진단을 조기 종료한다.
검증 episode에는 이 옵션을 쓰지 않는다. `expert` 모드의
`--omit-final-put`와 `--skip-action-index 8`은 음성 대조용이며, 원본 재생
결과와 분리해서 기록한다. `early_complete` 모드는 원본 초기 상태에서 허위
완료 요청이 목표 성공으로 오인되지 않는지 확인한다.

`events.jsonl`은 행동 전후 평가 상태, 원본 행동 변환, 시뮬레이터 내부 호출을
기록한다. RGB와 예측 mask는 PNG로, 실제 `/generate` 요청과 응답은 별도
JSON으로 저장하고 SHA-256으로 연결한다. 요청에는 RGB와 언어 지시만 들어간다.
객체 ID, segmentation,
평가기 상태는 정책 요청에 넣지 않는다. `runs/` 전체는 Git에서 무시되는
local-only artifact다. 추적되는 [tasks.json](tasks.json)과
[results.json](results.json)에 출처와 결과 요약을 남긴다.
