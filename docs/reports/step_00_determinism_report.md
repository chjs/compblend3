# Step 0 — HF eager forward 결정론 확인 보고서

## 1. 요약

HF transformers의 Mistral-7B-Instruct-v0.2 eager forward가 **같은 입력 + 같은 seed에서 bitwise 재현 가능**함을 확인했다. vast.ai A100-SXM4-80GB에서 fp32 / `attn_implementation="eager"` / PyTorch deterministic mode로 검증했고, **invariant 0.1·0.2 모두 통과** ✅. 이 결정론이 이후 모든 step의 invariant 검증의 토대가 된다.

local_a100 교차 검증은 이번 step에서 수행하지 않았다 — vast.ai 단독 invariant 검증으로 통과 판정 (사용자 결정, §6 참조).

## 2. 목표와 통과 기준

| Invariant | 명제 | 결과 |
|---|---|---|
| 0.1 | 같은 입력 + 같은 seed로 3회 forward → logits SHA-256 동일 | ✅ PASS |
| 0.2 | 다른 입력 → logits SHA-256 다름 | ✅ PASS |
| 0.3 | (best-effort) vast.ai ↔ local_a100 SHA-256 일치 또는 atol 1e-5 | ⬜ 미수행 (local_a100 검증 생략) |

`all_invariants_passed: true`.

## 3. 환경 정보

| 항목 | 값 |
|---|---|
| 실험 환경 | vast.ai A100-SXM4-80GB (instance 36769033, 검증 후 destroy) |
| GPU / driver | NVIDIA A100-SXM4-80GB / 525.125.06 |
| Python | 3.10.12 |
| PyTorch | 2.10.0+cu128 (CUDA 12.8) |
| transformers | 4.51.3 |
| 모델 | mistralai/Mistral-7B-Instruct-v0.2 (snapshot 63a8b081…) |
| dtype / attention | float32 / eager |
| 결정론 설정 | `use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` |

`scripts/check_env.py` (vastai 모드) 전 항목 통과 — `results/phase_00/vastai/env_check.json`.

## 4. 수정 / 신규 파일

| 파일 | 변경 | 사유 |
|---|---|---|
| `tasks/step_00/run_determinism_check.py` | 신규 | Step 0 결정론 검증 스크립트 (task 명세대로) |
| `scripts/vast_helper.py` | placeholder → 동작 코드 | 인스턴스 lifecycle 자동화. 이후 disk 필터·destroy 대화형 처리 수정 (§8) |
| `results/step_00/vastai/summary.json` | 신규 | 결정론 검증 결과 (vast.ai에서 생성) |
| `results/phase_00/vastai/env_check.json`, `env_info.txt` | 신규 | Phase 0-B 환경 검증 결과 |

메타 문서(`CLAUDE.md §7.1/§7.2`, `DECISIONS.md §9.6/§13`, `PROGRESS.md`)는 브랜치 워크플로우 보강으로 `main`에 별도 commit (`[meta]` prefix).

## 5. Tensor shape 명세

| 변수 | shape | dtype | 비고 |
|---|---|---|---|
| `logits` (prompt_A) | `(1, 6, 32000)` | float32 | B=1, T=6 ("The capital of France is" + BOS), vocab=32000 |
| SHA-256 입력 | `logits.cpu().to(float32).numpy().tobytes()` | — | bitwise 비교용 |

## 6. 구현 핵심

- **결정론 보장**: 모든 forward 직전 `set_all_seeds(42)` 호출 후 `torch.no_grad()`로 forward. `setup_deterministic()`은 프로그램 시작 시 1회.
- **SHA-256 해시**: logits를 fp32 numpy bytes로 변환 후 해시 — 환경 간 비교에서도 동일 기준 사용 가능.
- **invariant 0.1**: 같은 prompt_A 3회 forward의 SHA-256 집합 크기가 1인지 확인.
- **invariant 0.2**: prompt_B의 SHA-256이 prompt_A와 다른지 확인 — 0.1이 "모든 입력에 trivially 동일"이 아님을 보증.
- **last-token sanity**: prompt_A의 last-token top-1 id(5465)를 `tokenizer.decode`로 디코드 → `"Paris"`. invariant 0.2가 "출력이 garbage가 아닌 의미 있는 분포"임을 보강 (가중치·토크나이저 정상 확인).
- **KV cache 미사용**: `run_forward()`가 `model(..., use_cache=False)`로 호출 — plain forward. DynamicCache는 Step 2에서 도입.

## 7. Invariant 검증 결과

| Invariant | 검증 내용 | 결과 |
|---|---|---|
| 0.1 | run1/2/3 SHA-256 = `d338ec6a350dc6d7f81307c1479737cde13e7740d78ecf360b93ce4d23573c0b` (3회 동일) | ✅ PASS |
| 0.2 | prompt_B SHA-256 = `7cc4f432e2418b2a3410a1e51ab5fe09a60c22112ae964b24a085a442f70ba4d` ≠ prompt_A | ✅ PASS |

## 8. 결과 데이터 (`results/step_00/vastai/summary.json`)

| 필드 | 값 |
|---|---|
| `all_invariants_passed` | `true` |
| `logits_sha256` (대표) | `d338ec6a350dc6d7f81307c1479737cde13e7740d78ecf360b93ce4d23573c0b` |
| `logits_summary.shape` | `[1, 6, 32000]` |
| `logits_summary.norm` | `1751.87` |
| `logits_summary.max / min / mean` | `15.17 / -22.24 / -3.01` |
| `last_token_top5_indices` | `[5465, 264, 624, 2651, 272]` |
| `last_token_top1_decoded` | `"Paris"` — prompt_A "The capital of France is"의 top-1, 정상 ✅ (top-5 디코드: Paris / a / one / known / the) |

## 9. 환경 간 비교

local_a100 검증은 이번 step에서 **미수행** — 사용자 결정에 따라 vast.ai 단독으로 진행. `results/step_00/local_a100/`는 비어 있으며, `compare_results.py`는 vastai 단독 invariant 검증으로 통과 판정한다. 환경 간 SHA-256 일치(invariant 0.3)는 **사용자가 필요하다고 판단할 때 별도 round**로 수행한다.

## 10. 알려진 한계 / 의심스러운 부분

- ⚠️ **환경 간 결정론(0.3) 미검증**: vast.ai ↔ local_a100 SHA-256 일치 여부는 확인하지 않았다. cuDNN/driver 차이로 갈릴 수 있으며, 그 경우 atol 1e-5 fallback이 필요. Step 0 task 파일은 이 검증을 "강력 권장"으로 둠 — 보류 항목. **후속 영향**: Step N에서 환경 간 결과 불일치가 나오면 코드 버그와 환경 비결정성을 구분하기 어려워진다 — 그 문제가 실제로 발생하는 시점에 invariant 0.3 별도 round를 실행한다.
- 🔵 **driver 525.125.06**: 직전 인스턴스(570)보다 낮으나 `torch.cuda` / cu128 정상 동작 확인. troubleshooting.md의 "driver ≥ 525" 기준 충족.
- 🔵 **fp16/bf16 결정론**: 이번은 fp32 기준. lower precision은 결정론이 더 어려움 — Phase 6 (flash-attn 전환) 시 재확인 필요.
- 🔵 **인프라 이슈 2건 (셋업 중 발견·수정)**: ① 첫 인스턴스가 disk 작은 offer에 잡혀 `install_vastai.sh` "No space left" 실패 → `vast_helper.py`에 `disk_space>=100` 필터 + `setup_instance()` 디스크 가드 추가. ② `vastai destroy instance`가 `[y/N]` 대화형이라 첫 destroy가 rc=0이어도 실제로는 안 됨 → stdin "y" 전달 + 목록 검증 추가. 둘 다 코드로 수정 완료.

## 11. 다음 step

Step 1 — Our layerwise forward = HF 표준 forward (no cache). HF 표준 forward와 우리 layerwise forward의 logits 일치를 검증한다.
