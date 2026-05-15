"""compblend.modeling — HF modeling 파일의 무수정 fork (Step 1).

이 디렉토리의 코드는 transformers에서 fork한 그대로 — 수정 ❌
(CLAUDE.md §11-7, tasks/step_01_fork_equivalence.md "Step 1 원칙").
검증 계측은 외부 forward hook으로만 한다 (fork 디렉토리에 hook/assert 추가 ❌).
"""
from .modeling_mistral import MistralForCausalLM, MistralModel

__all__ = ["MistralForCausalLM", "MistralModel"]
