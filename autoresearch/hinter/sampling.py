"""The single shared RLAD sampler for hinted and unhinted student inference."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import REPO_ROOT, canonicalize_hint, verify_runtime
from .grader import grade_response, verify_grader


RL_CODE = REPO_ROOT / "train" / "rl"
if str(RL_CODE) not in sys.path:
    sys.path.insert(0, str(RL_CODE))

from eval.vllm_eval import sample_rendered_prompts  # noqa: E402
from rlad_plugin.templates import (  # noqa: E402
    render_prompt,
    render_prompt_with_abstraction,
)


@dataclass(frozen=True)
class SampledResponse:
    sample_idx: int
    response: str
    reward: int
    finish_reason: str | None
    completion_tokens: int


def require_one_slurm_gpu(torch_module: Any) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("student inference is forbidden outside Slurm")
    step = os.environ.get("SLURM_STEP_ID")
    if not step or not step.isdigit():
        raise RuntimeError("student inference requires a numeric Slurm step")
    visible = int(torch_module.cuda.device_count())
    if visible != 1:
        raise RuntimeError(
            f"student inference requires exactly one visible GPU, found {visible}"
        )


def sampling_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    sampling = config["sampling"]
    return {
        "n": int(sampling["rollouts"]),
        "temperature": float(sampling["temperature"]),
        "top_p": float(sampling["top_p"]),
        "top_k": int(sampling["top_k"]),
        "min_p": float(sampling["min_p"]),
        "presence_penalty": float(sampling["presence_penalty"]),
        "frequency_penalty": float(sampling["frequency_penalty"]),
        "repetition_penalty": float(sampling["repetition_penalty"]),
        "ignore_eos": bool(sampling["ignore_eos"]),
        "max_tokens": None,
        "seed": int(sampling["seed"]),
    }


def render_student_prompt(
    tokenizer: Any,
    problem: str,
    hint: str | None,
) -> str:
    if hint is None:
        return render_prompt(tokenizer, problem)
    return render_prompt_with_abstraction(
        tokenizer,
        problem,
        canonicalize_hint(hint),
    )


class StudentSampler:
    """One fixed Qwen3-1.7B engine, always confined to one Slurm GPU step."""

    def __init__(self, config: dict[str, Any]) -> None:
        import torch
        from transformers import AutoConfig, AutoTokenizer
        from vllm import LLM, SamplingParams

        require_one_slurm_gpu(torch)
        verify_runtime(config)
        verify_grader(config)
        student = config["student"]
        model_config = AutoConfig.from_pretrained(
            student["repo_id"],
            revision=student["revision"],
            trust_remote_code=True,
        )
        if int(model_config.max_position_embeddings) != int(
            student["max_model_len"]
        ):
            raise RuntimeError("student context-length drift")
        self.tokenizer = AutoTokenizer.from_pretrained(
            student["repo_id"],
            revision=student["revision"],
            trust_remote_code=True,
        )
        self.sampling_params_cls = SamplingParams
        self.config = config
        self.expected_rollouts = int(config["sampling"]["rollouts"])
        self.llm = LLM(
            model=student["repo_id"],
            revision=student["revision"],
            tokenizer_revision=student["revision"],
            tensor_parallel_size=1,
            seed=int(config["sampling"]["seed"]),
            max_model_len=int(student["max_model_len"]),
            gpu_memory_utilization=float(student["gpu_memory_utilization"]),
            trust_remote_code=True,
        )

    def sample_batch(
        self,
        requests: list[dict[str, str | None]],
    ) -> list[list[SampledResponse]]:
        if not requests:
            return []
        prompts = []
        for request in requests:
            problem = request["problem"]
            answer = request["answer"]
            if not isinstance(problem, str) or not isinstance(answer, str):
                raise TypeError("sampling requests need string problem/answer")
            prompts.append(
                render_student_prompt(
                    self.tokenizer,
                    problem,
                    request.get("hint"),
                )
            )
        outputs = sample_rendered_prompts(
            llm=self.llm,
            tokenizer=self.tokenizer,
            sampling_params_cls=self.sampling_params_cls,
            rendered_prompts=prompts,
            sampling_kwargs=sampling_kwargs(self.config),
            max_model_len=int(self.config["student"]["max_model_len"]),
        )
        result = []
        for request, output in zip(requests, outputs, strict=True):
            if len(output.outputs) != self.expected_rollouts:
                raise RuntimeError("vLLM returned the wrong rollout count")
            answer = request["answer"]
            assert isinstance(answer, str)
            group = []
            for sample_idx, completion in enumerate(output.outputs):
                group.append(
                    SampledResponse(
                        sample_idx=sample_idx,
                        response=completion.text,
                        reward=grade_response(completion.text, answer),
                        finish_reason=completion.finish_reason,
                        completion_tokens=len(completion.token_ids),
                    )
                )
            result.append(group)
        return result

    def sample(
        self,
        *,
        problem: str,
        answer: str,
        hint: str | None,
    ) -> list[SampledResponse]:
        return self.sample_batch(
            [{"problem": problem, "answer": answer, "hint": hint}]
        )[0]
