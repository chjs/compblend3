# LMCache CacheBlend — vLLM 수동 패치 (manual-insert)

> ⚠️ **이 파일은 `git apply` 불가다.**
> LMCache README의 패치 안내가 unified diff가 아니라 "수동 삽입용 Python 코드 블록"
> 형태이기 때문에, 확장자를 `.patch`가 아닌 `.md`로 둔다.
> vLLM 소스 트리에 **수동으로** 코드를 삽입해야 한다.

---

## 적용 대상

| 항목 | 값 |
|---|---|
| 수정 파일 | `vllm/v1/worker/gpu_worker.py` |
| 수정 함수 | `initialize_from_config(self, kv_cache_config: KVCacheConfig)` |
| 삽입 위치 | KV connector 초기화 **전** (함수 본문 시작부, docstring 직후) |
| vLLM 버전 | v0.17.1 (V1 Engine Alpha) |

## 진실 공급원 (source of truth)

| 항목 | 값 |
|---|---|
| LMCache repo | `chjs/LMCache` |
| Branch | `fix/cacheblend-vllm-v0.17.1-compat` |
| Commit SHA | `9f8aa4d6ee70a2a05657470f3b84d3298c05d8a1` |
| README raw URL | https://raw.githubusercontent.com/chjs/LMCache/fix/cacheblend-vllm-v0.17.1-compat/examples/blend_kv_v1/README.md |
| 확인 일자 | 2026-05-14 |
| 검증 방법 | `gh api -H "Accept: application/vnd.github.raw"` 와 `curl raw.githubusercontent.com` 두 방법으로 회수 → 바이트 단위 일치 (`diff` 결과 IDENTICAL, 길이 1686자) |

## README 원문 인용 (patch 섹션, 변형 없음)

아래 4-backtick 블록은 위 README 파일의 patch 섹션을 **원문 그대로** 옮긴 것이다
(줄바꿈/들여쓰기 보존, 내부 ` ```python ` 펜스 포함).

````
## Required ad-hoc changes in vLLM (v0.17.1)
To enable CacheBlend functionality, certain internal vLLM structures must be registered with LMCache during the worker initialization process.

Please apply the following changes to `vllm/v1/worker/gpu_worker.py`:

### Register Model Instance
In the function `initialize_from_config(self, kv_cache_config: KVCacheConfig)`, add the model registration logic before the KV connector is initialized.
This is required because LMCBlenderBuilder needs to access the model runner during connector setup.

```python
def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
    """Allocate GPU KV cache with the specified kv_cache_config."""

    # CacheBlend: register model with LMCache tracker before KV
    # connector init, because LMCBlenderBuilder.get_or_create() calls
    # VLLMModelTracker.get_model() during connector initialization.
    try:
        from lmcache.v1.compute.models.utils import VLLMModelTracker
        from lmcache.integration.vllm.utils import ENGINE_NAME
        VLLMModelTracker.register_model(
            ENGINE_NAME, self.model_runner.model)
    except ImportError:
        pass

    # Existing KV connector initialization follows...
```
````

## 적용 방법 (수동)

1. vLLM v0.17.1 소스 트리를 checkout (commit `95c0f928cdeeaa21c4906e73cee6a156e1b3b995`).
2. `vllm/v1/worker/gpu_worker.py`를 연다.
3. `initialize_from_config` 함수의 docstring 직후, 기존 KV connector 초기화 코드 **앞에**
   위 인용문의 `try/except ImportError` 블록(7줄)을 삽입한다.
4. `git apply`는 쓰지 않는다 — 위 코드 블록은 diff가 아니라
   "함수가 어떻게 보여야 하는지"의 예시다.

## 비고 — WebFetch 함정 기록

이 패치 정보를 처음 회수할 때 `WebFetch` 도구(작은 모델이 페이지를 재구성)는
원문에 **없던** 소제목(`## File Modified`, `## Code Block to Add`, `## Key Requirement`)을
임의로 만들어 붙였다. 핵심 내용(파일·함수·코드)은 정확했으나 섹션 구조가 왜곡됐다.
그래서 `gh api` + `curl` 두 방법으로 raw 원문을 재회수해 바이트 일치를 확인한 뒤
이 파일을 작성했다.

**교훈: 외부 문서를 reference로 pinning할 때는 raw 원문 회수 + 교차검증을 거친다.**
