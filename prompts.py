REFLECTION_META_PROMPT = """I gave an assistant the following instruction to perform a task:

<current_instruction>
{old_prompt}
</current_instruction>

Below are recent attempts. For each: what the assistant received as input,
what it produced, the score it got, and feedback on what went wrong or right.

{trace_blocks}

Diagnose the systematic issues with the current instruction based on these traces.
Then write a NEW instruction that:
- preserves what is working
- fixes the recurring failure modes evidenced in the feedback
- stays a self-contained instruction (the assistant only sees the new instruction
  plus its input, NOT these examples)
- keeps the {{task}} placeholder so the harness can still inject the task

Return ONLY the new instruction text wrapped in <new_instruction></new_instruction> tags."""

TRACE_BLOCK = """── Attempt {n} ──
Input: {input}
Output: {output}
Score: {score:.2f}
Feedback: {feedback}
"""