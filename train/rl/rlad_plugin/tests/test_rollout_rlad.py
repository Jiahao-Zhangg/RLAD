"""Unit tests for the RLAD-Hierarchical on-policy rollout (rollout_rlad.generate_hierarchical) and the RLAD-Hierarchical reward
post-processor (reward_post.hierarchical_reward_post). Pure-python, no GPU / no network:
  - _child / _extract_hint_text / the (1+m)-strided index scheme (pure helpers);
  - full generate_hierarchical with `post` + tokenizer stubbed → assert sample assembly + metadata;
  - hierarchical_reward_post on a synthetic flat sample list → assert hint = normalized mean of its sols.

Run (host env, both plugin + miles on PYTHONPATH):
  cd train/rl
  PYTHONPATH=$PWD:$PWD/miles python -m pytest rlad_plugin/tests/test_rollout_rlad.py -q
"""

import asyncio
import math
from types import SimpleNamespace

from miles.rollout.base_types import GenerateFnInput
from miles.utils.types import Sample

from rlad_plugin import rollout_rlad
from rlad_plugin.reward_post import _grpo_norm, hierarchical_reward_post


# --------------------------------------------------------------------------- stubs
class _StubTok:
    """Minimal tokenizer: apply_chat_template returns the user content; encode → int ids."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, tools=None, **kw):
        assert tokenize is False
        return "PROMPT::" + msgs[-1]["content"]

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


def _args(**over):
    d = dict(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=1234,
        rollout_max_response_len=16384,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        sglang_speculative_algorithm=None,
        partial_rollout=False,
    )
    d.update(over)
    return SimpleNamespace(**d)


def _canned(text):
    # mimics SGLang /generate: 3 (logprob, token_id) pairs + a normal stop
    return {"text": text, "meta_info": {"output_token_logprobs": [[-0.1, 5], [-0.2, 6], [-0.3, 7]],
                                        "finish_reason": {"type": "stop"}}}


def _parent(qid="pope-9", index=7, problem="What is 2+2?", label="4"):
    return Sample(group_index=3, index=index, prompt="orig-plain-solver-prompt", label=label,
                  metadata={"qid": qid, "problem": problem, "mean_reward": 0.7, "sources": ["s"]})


# --------------------------------------------------------------------------- pure helpers
def test_child_clears_generation_and_isolates_metadata():
    p = _parent()
    c = rollout_rlad._child(p, prompt="NEWP", index=42, md_updates={"mrt_kind": "hint", "hint_id": "h0"})
    assert c.tokens == [] and c.response == "" and c.response_length == 0 and c.reward is None
    assert c.loss_mask is None
    assert c.status == Sample.Status.PENDING
    assert c.prompt == "NEWP" and c.index == 42
    assert c.group_index == p.group_index and c.label == p.label      # identity kept
    assert c.metadata["qid"] == "pope-9" and c.metadata["mrt_kind"] == "hint" and c.metadata["hint_id"] == "h0"
    # mutating the child's metadata must not touch the parent's
    c.metadata["mrt_kind"] = "solution"
    assert p.metadata.get("mrt_kind") is None
    assert "mrt_kind" not in p.metadata


def test_extract_hint_text():
    assert rollout_rlad._extract_hint_text("a</think>b") == "b"
    assert rollout_rlad._extract_hint_text("x</think>y</think>z") == "z"
    assert rollout_rlad._extract_hint_text("  c  ") == "c"
    assert rollout_rlad._extract_hint_text("") == ""


def test_index_scheme_disjoint_monotonic():
    m = 4
    m_plus = 1 + m
    for i in (0, 1, 2, 7):
        base = i * m_plus
        idxs = [base] + [base + 1 + j for j in range(m)]
        assert idxs == list(range(i * m_plus, i * m_plus + m_plus))
        # block i strictly precedes block i+1 (keeps qid group contiguous after flatten)
        assert idxs[-1] < (i + 1) * m_plus


# --------------------------------------------------------------------------- full generate
def test_generate_hierarchical_assembly(monkeypatch):
    m = 3
    calls = []

    async def fake_post(url, payload):
        calls.append(payload)
        # first call is the hint, the rest are solutions
        return _canned("HINT </think> - insight" if len(calls) == 1 else "sol </think> \\boxed{4}")

    monkeypatch.setattr(rollout_rlad, "post", fake_post)
    monkeypatch.setenv("RLAD_SOLUTIONS_PER_HINT", str(m))   # m is read from env (add_arguments unwired here)
    inp = GenerateFnInput(
        state=SimpleNamespace(tokenizer=_StubTok(), processor=None, args=_args()),
        sample=_parent(index=2),
        sampling_params={"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 16384},
        evaluation=False,
    )
    out = asyncio.run(rollout_rlad.generate_hierarchical(inp))
    samples = out.samples
    assert isinstance(samples, list) and len(samples) == 1 + m

    hint = samples[0]
    assert hint.metadata["mrt_kind"] == "hint"
    assert hint.reward == 0.0                          # set → RM skipped
    assert hint.metadata["hint_id"] == "pope-9#2"
    assert hint.index == 2 * (1 + m)                   # base
    for j, sol in enumerate(samples[1:]):
        assert sol.metadata["mrt_kind"] == "solution"
        assert sol.reward is None                      # left None → miles RM grades it
        assert sol.metadata["hint_id"] == "pope-9#2"   # points at its parent hint
        assert sol.metadata["qid"] == "pope-9"
        assert sol.index == 2 * (1 + m) + 1 + j
        assert sol.loss_mask is None                   # → all-ones (full response trained)
    # hint got the short budget; solutions got the full budget
    assert calls[0]["sampling_params"]["max_new_tokens"] == 1024
    assert calls[1]["sampling_params"]["max_new_tokens"] == 16384
    # hint prompt from absgen template, solution prompt carries the <cheatsheet>
    assert "Propose a concise cheatsheet" in hint.prompt
    assert "<cheatsheet>" in samples[1].prompt


def test_generate_hierarchical_aborted_remapped(monkeypatch):
    async def fake_post(url, payload):
        out = _canned("x </think> y")
        out["meta_info"]["finish_reason"] = {"type": "abort"}   # would make miles skip RM for the whole list
        return out

    monkeypatch.setattr(rollout_rlad, "post", fake_post)
    monkeypatch.setenv("RLAD_SOLUTIONS_PER_HINT", "2")
    inp = GenerateFnInput(
        state=SimpleNamespace(tokenizer=_StubTok(), processor=None, args=_args()),
        sample=_parent(),
        sampling_params={"max_new_tokens": 16384},
        evaluation=False,
    )
    out = asyncio.run(rollout_rlad.generate_hierarchical(inp))
    assert all(s.status != Sample.Status.ABORTED for s in out.samples)
    assert all(s.status == Sample.Status.TRUNCATED for s in out.samples)


# --------------------------------------------------------------------------- reward post
def _s(qid, kind, hint_id, reward):
    return Sample(metadata={"qid": qid, "mrt_kind": kind, "hint_id": hint_id}, reward=reward)


def test_hierarchical_reward_post_hint_is_normalized_mean_of_its_solutions():
    # one prompt: 2 hints × 4 solutions; flat order = [h0, s0..s3, h1, s0..s3]
    samples = [
        _s("pope-1", "hint", "h0", 0.0),
        _s("pope-1", "solution", "h0", 1.0), _s("pope-1", "solution", "h0", 1.0),
        _s("pope-1", "solution", "h0", 0.0), _s("pope-1", "solution", "h0", 0.0),
        _s("pope-1", "hint", "h1", 0.0),
        _s("pope-1", "solution", "h1", 0.0), _s("pope-1", "solution", "h1", 0.0),
        _s("pope-1", "solution", "h1", 0.0), _s("pope-1", "solution", "h1", 1.0),
    ]
    raw, adv = hierarchical_reward_post(SimpleNamespace(), samples)

    # solutions: grpo-norm over all 8 correctness values, in order
    sol_vals = [1, 1, 0, 0, 0, 0, 0, 1]
    exp_sol = _grpo_norm(sol_vals)
    sol_idx = [1, 2, 3, 4, 6, 7, 8, 9]
    for k, i in enumerate(sol_idx):
        assert math.isclose(adv[i], exp_sol[k], rel_tol=1e-9, abs_tol=1e-9)

    # hints: each = mean of its 4 solutions, then grpo-norm over the 2 hints
    exp_hint = _grpo_norm([0.5, 0.25])            # mean([1,1,0,0]), mean([0,0,0,1])
    assert math.isclose(adv[0], exp_hint[0], rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(adv[5], exp_hint[1], rel_tol=1e-9, abs_tol=1e-9)
    assert adv[0] > adv[5]                          # hint h0 (better sols) ranked above h1
    assert raw[0] == 0.0 and raw[5] == 0.0          # hint raw reward ignored by the aggregation


def test_hierarchical_reward_post_groups_split_by_qid():
    # two prompts back-to-back must be normalized INDEPENDENTLY (contiguous-qid grouping)
    samples = [
        _s("pope-1", "hint", "a0", 0.0), _s("pope-1", "solution", "a0", 1.0), _s("pope-1", "solution", "a0", 1.0),
        _s("pope-2", "hint", "b0", 0.0), _s("pope-2", "solution", "b0", 1.0), _s("pope-2", "solution", "b0", 0.0),
    ]
    _, adv = hierarchical_reward_post(SimpleNamespace(), samples)
    # pope-1 solutions both correct → identical (mean-subtracted) advantage 0.0
    assert math.isclose(adv[1], adv[2], abs_tol=1e-9) and math.isclose(adv[1], 0.0, abs_tol=1e-9)
    # pope-2 solutions differ → nonzero, opposite-sign advantages
    assert adv[4] > 0 > adv[5]
