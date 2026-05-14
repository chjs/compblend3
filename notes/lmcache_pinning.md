# LMCache Reference Pinning

> Phase 7(vLLM/LMCache reference 실행)에서 쓸 reference를 고정한다.
> LMCache repo는 빠르게 변하므로 commit SHA로 핀을 박는다.
> 관련 결정: DECISIONS.md §7.2

확정일: 2026-05-14
확정 방법: `gh api` + `curl` 두 방법으로 raw 원문 회수 → 바이트 일치 확인

---

## LMCache

| 항목 | 값 |
|---|---|
| Repo | `chjs/LMCache` |
| Branch | `fix/cacheblend-vllm-v0.17.1-compat` |
| Commit SHA | `9f8aa4d6ee70a2a05657470f3b84d3298c05d8a1` |
| Commit date | 2026-04-13T05:45:37Z |
| Commit message | `Merge branch 'dev' into fix/cacheblend-vllm-v0.17.1-compat` |
| Branch 확인 URL | https://github.com/chjs/LMCache/tree/fix/cacheblend-vllm-v0.17.1-compat |
| Pinned tree URL | https://github.com/chjs/LMCache/tree/9f8aa4d6ee70a2a05657470f3b84d3298c05d8a1 |
| README raw URL | https://raw.githubusercontent.com/chjs/LMCache/fix/cacheblend-vllm-v0.17.1-compat/examples/blend_kv_v1/README.md |

## vLLM

| 항목 | 값 |
|---|---|
| 버전 | v0.17.1 (V1 Engine Alpha) — LMCache README 원문에 명시 |
| Commit SHA | `95c0f928cdeeaa21c4906e73cee6a156e1b3b995` |
| 확인 방법 | `gh api repos/vllm-project/vllm/git/ref/tags/v0.17.1` → lightweight tag, `object.type=commit` (annotated tag 아님, commit 직결) |
| Tag 확인 URL | https://github.com/vllm-project/vllm/releases/tag/v0.17.1 |

## Patch

| 항목 | 값 |
|---|---|
| 파일 | `patches/lmcache-vllm-cacheblend.md` (확장자 `.patch` ❌) |
| 형식 | **수동 삽입용 Python 코드 블록** — unified diff 아님, `git apply` 불가 |
| 적용 대상 | vLLM 소스 트리: `vllm/v1/worker/gpu_worker.py` |
| 적용 함수 | `initialize_from_config(self, kv_cache_config: KVCacheConfig)` |
| 적용 방법 | 위 함수의 docstring 직후, KV connector 초기화 코드 앞에 README의 `try/except ImportError` 블록(7줄)을 수동 삽입 |

**형식 판단 근거**: LMCache README의 patch 섹션에 `@@`, `---`, `+++`, `+`/`-` 같은
unified diff 마커가 전혀 없고, ` ```python ` 코드 블록으로 "함수가 어떻게 보여야 하는지"를
보여주는 형태였음. 그래서 `.patch`가 아닌 `.md`로 저장.
상세 원문 인용은 `patches/lmcache-vllm-cacheblend.md` 참조.

## 검증 계획

- 위 SHA들로 checkout 후 LMCache README 예제(`examples/blend_kv_v1/blend.py`)가
  실행 가능한지는 **Phase 7 진입 시 사용자가 직접 확인**한다.
- 본 Phase 0-D는 정보 회수 + pinning만 수행 (실행 ❌).

## 정직성 노트

- 1차 회수에 쓴 `WebFetch` 도구는 원문에 없던 소제목을 임의로 추가했다.
  `gh api` + `curl` raw 원문 교차검증으로 바로잡았다. (`patches/lmcache-vllm-cacheblend.md` 비고 참조)
- vLLM commit SHA는 v0.17.1 tag가 가리키는 commit이다. LMCache README가
  "tested and verified with vLLM v0.17.1"이라 명시했으므로 이 SHA를 reference로 핀.
