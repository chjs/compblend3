# patches/

외부 reference 저장소에 적용하는 patch 파일들.

## 현재 patches

| 파일 | 대상 | 적용 시점 |
|---|---|---|
| `lmcache-vllm-cacheblend.patch` | vllm 소스 트리 | Phase 0-D에서 회수, Phase 7에서 적용 |

(Phase 0-D 완료 전까지는 비어 있음 — placeholder.)

## 적용 방법

```bash
cd /path/to/vllm
git apply /path/to/compblend3/patches/<patch_name>.patch
```

## 회수 출처

각 patch의 원본 위치 (어느 README의 어느 섹션)는 `notes/lmcache_pinning.md` 등 관련 메모에 명시.
