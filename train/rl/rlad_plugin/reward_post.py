"""Reward post-processing for RLAD.

Hook: `--custom-reward-post-process-path`. Called f(args, samples) on the FLAT,
group-contiguous list of all rollout samples; returns (raw_rewards, processed). The custom
function REPLACES miles' default group normalization, and with --advantage-estimator grpo
the returned `processed[i]` is broadcast as sample i's per-token advantage (KL is a SEPARATE
loss via --use-kl-loss, so zeroing the advantage still keeps the KL regularizer). Verified:
miles/ray/rollout/train_data_conversion.py:_post_process_rewards + advantages.py:get_grpo_returns.

Two entry points:

- rlad_reward_post (RLAD solution-generator DAPO): per the paper's Eq. 3, reward-MASK the
  no-abstraction prompts. For each prompt group: if it is abstraction-conditioned, standard
  GRPO group-normalize (r-mean)/(std+eps); if it is a bare (no-abstraction) prompt, set every
  advantage to 0 (the KL term still pulls it toward the reference). Group membership and the
  has_abstraction flag come from sample.metadata (stamped by data_prep / rollout_rlad).

- hierarchical_reward_post (variant RLAD-Hierarchical, hint-then-solutions GRPO, INT-002 extension): per prompt, the
  rollout emits n hint samples (mrt_kind='hint') and, for each hint, m solution samples
  (mrt_kind='solution', md['hint_id']). Solutions are rewarded by correctness and
  group-normalized across all n*m solutions of the prompt; each hint's advantage = the MEAN
  raw reward of ITS m solutions, group-normalized across the n hints. (Single model, one RL run.)
"""

from statistics import mean, stdev

_EPS = 1e-6


def _groups_by(samples, keyfn):
    """Contiguous spans sharing keyfn(sample) (groups are contiguous in the flat list)."""
    spans, start = [], 0
    for i in range(1, len(samples) + 1):
        if i == len(samples) or keyfn(samples[i]) != keyfn(samples[start]):
            spans.append((start, i))
            start = i
    return spans


def _md(s):
    return s.metadata if isinstance(s.metadata, dict) else {}


def _gkey(s):
    return _md(s).get("qid", s.group_index)


def _grpo_norm(vals):
    m = mean(vals)
    # sample std (ddof=1) to MATCH miles' default group-norm (torch.std, unbiased=True) used by
    # the baseline arm — keeps the RLAD vs +DAPO A/B on an identical advantage scale at n=16
    # (preflight audit w8okt63ib: pstdev/ddof=0 would make RLAD advantages ~3% larger).
    sd = stdev(vals) if len(vals) > 1 else 0.0
    return [(v - m) / (sd + _EPS) for v in vals]


def rlad_reward_post(args, samples):
    """RLAD sol-gen: GRPO-normalize abstraction groups; zero advantage on no-abstraction groups."""
    raw = [float(s.reward) for s in samples]
    adv = [0.0] * len(samples)
    for a, b in _groups_by(samples, _gkey):
        idxs = list(range(a, b))
        has_abs = bool(_md(samples[a]).get("has_abstraction", False))  # fail-safe: missing flag → mask (Eq.3)
        if not has_abs:
            continue  # zeroed advantage; KL retained via --use-kl-loss
        adv_g = _grpo_norm([raw[i] for i in idxs])
        for j, i in enumerate(idxs):
            adv[i] = adv_g[j]
    return raw, adv


def hierarchical_reward_post(args, samples):
    """RLAD-Hierarchical: hints get the group-normalized mean-of-downstream-solution reward; solutions get
    the group-normalized correctness reward (normalized across the prompt's n*m solutions)."""
    raw = [float(s.reward) for s in samples]
    adv = [0.0] * len(samples)
    for a, b in _groups_by(samples, _gkey):  # one prompt
        idxs = list(range(a, b))
        hints = [i for i in idxs if _md(samples[i]).get("mrt_kind") == "hint"]
        sols = [i for i in idxs if _md(samples[i]).get("mrt_kind") == "solution"]
        # solution advantages: group-norm correctness across all solutions of this prompt
        if sols:
            sa = _grpo_norm([raw[i] for i in sols])
            for j, i in enumerate(sols):
                adv[i] = sa[j]
        # hint advantage = mean raw reward of its m downstream solutions, then group-norm over hints
        hint_means = []
        for hi in hints:
            hid = _md(samples[hi]).get("hint_id")
            downstream = [raw[i] for i in sols if _md(samples[i]).get("hint_id") == hid]
            hint_means.append(mean(downstream) if downstream else 0.0)
        if hints:
            ha = _grpo_norm(hint_means)
            for j, i in enumerate(hints):
                adv[i] = ha[j]
    return raw, adv
