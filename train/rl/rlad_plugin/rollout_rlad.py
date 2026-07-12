"""Custom on-policy rollout functions for the RLAD reproduction.

INT-002 extension RLAD-Hierarchical (hint-then-solutions GRPO), single-model, fully on-policy — the
paper's Eq. 3 reward realized inside one RL run (no offline abstraction scoring).

`generate_hierarchical` is invoked once per prompt-copy (miles makes `--n-samples-per-prompt`=n
deep-copies per prompt, one generate() call each). EACH call produces ONE hint + m solutions
conditioned on that hint, returned as a (1+m)-list of Samples. So a prompt yields n hints and
n*m solutions. Metadata stamped per sample:
  - qid       : prompt group key (from the source row's metadata; also used by hierarchical_reward_post)
  - mrt_kind  : "hint" | "solution"
  - hint_id   : unique per hint; each solution carries its parent hint's id
`rlad_plugin.reward_post.hierarchical_reward_post` (already implemented) consumes this: solutions get
GRPO-normalized correctness over the prompt's n*m solutions; each hint gets the group-normalized
MEAN raw reward of ITS m solutions. Advantage (not loss_mask) is zeroed to drop PG while keeping
the per-token KL, so hint+solution tokens are all trained (loss_mask left None -> all-ones).

Mirrors miles' stock single-turn POST core (generate_hub/single_turn.py) looped like a
multi-sample return. Wire via the config:
  --custom-generate-function-path rlad_plugin.rollout_rlad.generate_hierarchical
  --custom-reward-post-process-path rlad_plugin.reward_post.hierarchical_reward_post
  --solutions-per-hint m --hint-max-tokens H

RLAD-Joint is NOT here: it is the STOCK rollout on a combined-instruction prompt
(one generation containing <cheatsheet> then solution, whole-trajectory reward, plain GRPO) —
see configs/rlad_joint.sh + templates.render_joint_prompt.

m (solutions/hint) and the hint token cap are read from the ENV VARS RLAD_SOLUTIONS_PER_HINT /
RLAD_HINT_MAX_TOKENS (defaults 4 / 1024). miles only wires a custom generate fn's `add_arguments`
when the experimental rollout refactor is enabled (arguments.py:1826) — it is OFF here — so CLI
flags like --solutions-per-hint are rejected; jobs/launch_train.sh forwards these two vars into the
ray-worker RUNTIME_ENV_JSON (the same way it forwards other worker env), and rlad_hierarchical.sh sets M_SOLS there.
KEEP m consistent with the config's N_HINTS*(1+m) global-batch, or the rollout trim splits a group.
"""

import os
from copy import deepcopy

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import (  # generate_utils is a SIBLING of generate_hub
    compute_prompt_ids_from_sample,
    compute_request_payload,
    update_sample_from_response,
)
from miles.utils.http_utils import post
from miles.utils.types import Sample

from rlad_plugin.templates import render_absgen_prompt, render_prompt_with_abstraction

_DEFAULT_SOLUTIONS_PER_HINT = 4
_DEFAULT_HINT_MAX_TOKENS = 1024


def _int_env(key: str, default: int) -> int:
    """Env-var int with default; treats unset/empty (arms that don't set it) as default."""
    v = os.environ.get(key)
    return int(v) if v not in (None, "") else default


# ---- pure helpers (unit-tested without GPU / network) ------------------------
def _child(parent: Sample, *, prompt: str, index: int, md_updates: dict) -> Sample:
    """A fresh generation child of the prompt-sample: keep identity (group_index, label),
    clear every generation field, set a new prompt/index, and give it its OWN metadata dict
    (never aliasing the parent's)."""
    s = deepcopy(parent)
    s.reset_for_retry()  # clears tokens/response/response_length/logprobs/loss_mask/reward
    s.status = Sample.Status.PENDING  # reset_for_retry leaves ABORTED; single-turn core needs PENDING/ABORTED
    s.prompt = prompt  # reset_for_retry KEEPS parent.prompt; overwrite with the rendered prompt
    s.index = index
    s.metadata = {**(parent.metadata or {}), **md_updates}
    return s


def _extract_hint_text(resp: str) -> str:
    """render_absgen_prompt is enable_thinking=False (normally no <think>); strip defensively."""
    if not resp:
        return ""
    return resp.split("</think>")[-1].strip() if "</think>" in resp else resp.strip()


# ---- network core (mirrors single_turn.generate; fresh samples only) ---------
async def _one_post(state, args, sample: Sample, sampling_params: dict) -> Sample:
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    prompt_ids = compute_prompt_ids_from_sample(state, sample)  # tokenizes sample.prompt (a str)
    payload, halt_status = compute_request_payload(
        args, input_ids=prompt_ids, sampling_params=sampling_params, multimodal_inputs=sample.multimodal_inputs
    )
    if payload is None:  # prompt longer than the token budget
        sample.status = halt_status
        return sample
    output = await post(url, payload)
    await update_sample_from_response(args, sample, payload=payload, output=output)  # loss_mask stays None -> all-ones
    return sample


async def generate_hierarchical(input: GenerateFnInput) -> GenerateFnOutput:
    state = input.state
    args = input.args
    parent = input.sample
    assert not getattr(args, "partial_rollout", False), "generate_hierarchical does not support partial rollout"

    m = _int_env("RLAD_SOLUTIONS_PER_HINT", _DEFAULT_SOLUTIONS_PER_HINT)
    hint_cap = _int_env("RLAD_HINT_MAX_TOKENS", _DEFAULT_HINT_MAX_TOKENS)
    m_plus = 1 + m

    md = parent.metadata or {}
    problem = md["problem"]  # RAW problem text (train_pope_hard.jsonl stamps this)
    qid = md.get("qid", parent.group_index)
    hint_id = f"{qid}#{parent.index}"  # unique hint within the prompt group
    base = parent.index * m_plus  # (1+m)-strided block -> globally disjoint & monotonic across copies

    # (1) hint --------------------------------------------------------------
    hint = _child(
        parent,
        prompt=render_absgen_prompt(state.tokenizer, problem),
        index=base,
        md_updates={"qid": qid, "mrt_kind": "hint", "hint_id": hint_id},
    )
    await _one_post(state, args, hint, {**input.sampling_params, "max_new_tokens": hint_cap})
    hint.reward = 0.0  # skip RM (reward is not None); hierarchical_reward_post recomputes from the m solutions
    hint_text = _extract_hint_text(hint.response)

    # (2) m solutions conditioned on the hint -------------------------------
    samples = [hint]
    for j in range(m):
        sol = _child(
            parent,
            prompt=render_prompt_with_abstraction(state.tokenizer, problem, hint_text),
            index=base + 1 + j,
            md_updates={"qid": qid, "mrt_kind": "solution", "hint_id": hint_id},
        )
        await _one_post(state, args, sol, {**input.sampling_params})  # full max_new_tokens
        # sol.reward stays None -> batched_async_rm grades it via reward_math.custom_rm
        samples.append(sol)

    # (3) an ABORTED sample makes generate_and_rm skip RM for the WHOLE list -> reward stays None
    #     -> float(None) crash in hierarchical_reward_post. Remap to TRUNCATED (still graded, ~0 reward).
    for s in samples:
        if s.status == Sample.Status.ABORTED:
            s.status = Sample.Status.TRUNCATED

    return GenerateFnOutput(samples=samples)
