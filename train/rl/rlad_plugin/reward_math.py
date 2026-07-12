"""Training reward: 0/1 math correctness for the RLAD reproduction.

Wraps miles' built-in deepscaler grader (the same grader the offline eval harness
uses, so train-reward == eval-metric semantics). The grader looks only at the text
after the LAST "</think>", extracts the last \\boxed{...}, and checks math
equivalence vs the label.

miles calls a custom RM two ways (miles/rollout/rm_hub/__init__.py):
  async_rm(args, sample)          -> custom_rm(args, sample)
  batched_async_rm(args, samples) -> custom_rm(args, samples)   (batch mode)
so custom_rm must accept Sample | list[Sample].

For RLAD the abstraction lives in the PROMPT (solution generator) or earlier in the
SAME trajectory (variant RLAD-Joint hint->solution); either way the final solution ends with a
normal "</think> ... \\boxed{}" so the stock grader applies directly. metadata["mrt_kind"]
is unused here; reward shaping (reward-masking no-abstraction prompts; RLAD-Hierarchical hint-level
aggregation) happens in reward_post.py, not here.
"""

from miles.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward
from miles.utils.types import Sample


def _score_one(sample: Sample) -> float:
    response = sample.response or ""
    label = sample.label
    if label is None or not response.strip():
        return 0.0
    try:
        return float(get_deepscaler_rule_based_reward(response, label))
    except Exception:
        return 0.0


async def custom_rm(args, sample, **kwargs):
    if isinstance(sample, list):
        return [_score_one(s) for s in sample]
    return _score_one(sample)
