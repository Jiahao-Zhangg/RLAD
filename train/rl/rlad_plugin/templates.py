"""Prompt templates for the RLAD reproduction.

Written from the paper / method_spec only (the exact prompts are NOT in the paper —
logged assumptions; see ../../paper/method_spec.md "[not stated in paper]"):

- Solution generator conditioning: the paper (Fig. 4) frames the abstraction to the
  solver as a "cheatsheet". We inject the abstraction(s) in a <cheatsheet>...</cheatsheet>
  block placed before the problem, with an instruction to use it but not assume it is
  complete. (assumption A-cond)
- Boxed-answer instruction appended so the deepscaler grader can extract \\boxed{}.
  (assumption A-eval-prompt)
- Qwen3 thinking format: apply the chat template (enable_thinking default True) and
  ensure generation starts inside "<think>\\n" (the deepscaler grader splits on
  "</think>"). No system prompt.
"""

THINK_OPEN = "<think>\n"

BOXED_INSTRUCTION = (
    "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
)

# How the abstraction ("hint" / "cheatsheet") is shown to the solution generator.
CHEATSHEET_HEADER = (
    "Here is a cheatsheet of useful insights, strategies, or cautions for this problem. "
    "It does not contain the final answer; use whatever is helpful and ignore the rest.\n"
)


def _ensure_think_open(rendered: str) -> str:
    """Normalize so the prompt ends with exactly one '<think>\\n' (Qwen3 templates
    differ on whether they pre-append it after the assistant tag)."""
    if not rendered.rstrip().endswith("<think>"):
        return rendered + THINK_OPEN
    if not rendered.endswith("\n"):
        return rendered + "\n"
    return rendered


def render_prompt(tokenizer, question: str) -> str:
    """w/o-abstraction solution prompt: problem only (base / DAPO / w/o-abs eval)."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": question + BOXED_INSTRUCTION}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return _ensure_think_open(rendered)


def render_prompt_with_abstraction(tokenizer, question: str, abstraction: str) -> str:
    """w/-abstraction solution prompt: cheatsheet block + problem (RLAD solver)."""
    user = (
        CHEATSHEET_HEADER
        + "<cheatsheet>\n"
        + abstraction.strip()
        + "\n</cheatsheet>\n\n"
        + question
        + BOXED_INSTRUCTION
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return _ensure_think_open(rendered)


# ----- abstraction generator (pi_abs) prompt -------------------------------------
# The abstraction generator is asked to PROPOSE an abstraction/hint for a problem
# WITHOUT solving it or revealing the answer (method_spec; leakage constraint).
ABSGEN_INSTRUCTION = (
    "You are given a competition math problem. Propose a concise cheatsheet of useful "
    "procedural and factual insights (key ideas, lemmas, strategies, or common pitfalls) "
    "that would help someone solve it. Do NOT solve the problem and do NOT reveal the "
    "final answer. Keep it to a few focused bullet points."
)


def render_absgen_prompt(tokenizer, question: str) -> str:
    """Prompt for pi_abs to propose an abstraction for a problem (no answer leak).

    Direct (non-thinking) output: pi_abs emits the cheatsheet straight away — abstraction
    proposal is a short generation, and the SFT label is the cheatsheet text itself
    (A-absgen-format). enable_thinking=False where the template supports it; otherwise we
    do NOT pre-fill <think>."""
    msgs = [{"role": "user", "content": ABSGEN_INSTRUCTION + "\n\nProblem:\n" + question}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ----- RLAD-Joint prompt ---------------------------------------------
# INT-002 extension RLAD-Joint: a SINGLE model, in ONE thinking trajectory, first writes a cheatsheet
# then solves using it; the whole trajectory gets the solution's correctness reward (plain
# GRPO, stock rollout). The cheatsheet is produced inside the same generation (thinking ON),
# so no separate hint call — contrast with pi_abs (render_absgen_prompt, thinking off).
JOINT_INSTRUCTION = (
    "First, inside a <cheatsheet>...</cheatsheet> block, write a concise cheatsheet of useful "
    "insights, strategies, or cautions for the problem below (key ideas, lemmas, or common "
    "pitfalls). Do NOT put the final answer in the cheatsheet. Then, using your cheatsheet, "
    "solve the problem."
)


def render_joint_prompt(tokenizer, question: str) -> str:
    """RLAD-Joint prompt: propose a cheatsheet then solve, in one generation."""
    user = JOINT_INSTRUCTION + "\n\nProblem:\n" + question + BOXED_INSTRUCTION
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return _ensure_think_open(rendered)
