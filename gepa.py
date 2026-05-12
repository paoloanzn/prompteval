from typing import Any
from dataclasses import dataclass
from collections.abc import Callable
from evaluation import (
    chat,
    add_user_message,
    add_assistant_message,
    compile_prompt_template,
    generate_dataset,
    run_prompt,
    grade_by_model,
)
from prompts import REFLECTION_META_PROMPT, TRACE_BLOCK

type Schema = dict[Any, Any]

# checks key:value shape's matching between two Schema
def matches_schema(value: Schema, schema: Schema) -> bool:
    # example:
    # schema = {"name": str, "age": int}
    # value  = {"name": "john", "age": 30}
    return all(
        key in value and isinstance(value[key], expected_type)
        for key, expected_type in schema.items()
    )

# right now we support just one-module only systems with the specific in and out schemas
# { system_prompt: str, task: str } -> x
# { response: str } -> y

# S = (M, C, X, Y)
# S(x -> instance of Schema) returns y instance of Schema
# x must match S.in_schema and y must match S.out_schema
@dataclass
class System:
    modules: list[Module]
    control_flow: Callable[[list[Module], Schema], tuple[Schema, list[Schema]]]
    in_schema: Schema
    out_schema: Schema

    def __call__(self, x: Schema) -> list[Schema, list[Schema]]:
        if not matches_schema(x, self.in_schema):
            raise ValueError("Input does not match system input schema")

        y, traces = self.control_flow(self.modules, x)

        if not matches_schema(y, self.out_schema):
            raise ValueError("Output does not match system output schema")

        return y, traces

# Mi = (Pi, Wi, Xi, Yi) 
# Wi, model weights -> not relevant for GEPA
@dataclass
class Module:
    prompt: str # Pi -> mutated
    in_schema: Schema
    out_schema: Schema
    run_inference: Callable[[Schema, str], tuple[Schema, Schema]]
    
    def __call__(self, x: Schema) -> Schema:
        if not matches_schema(x, self.in_schema):
            raise ValueError("Input does not match module input schema")

        y, traces = self.run_inference(x, self.prompt)

        if not matches_schema(y, self.out_schema):
            raise ValueError("Output does not match module output schema")

        return y, traces

# Bundle of prompts of every module Mi in the system S
class Candidate:
    prompts: list[str]
    parent_index: int | None = None

# what the system does not see that is needed to score the output y in shape Y of it
class InstanceMetadata:
    gold: dict[str, Any]


# takes as input 
# - the output y of a system S
# - a grading prompt to evaluate it
# - m metadata for the the scoring
# -> returns a score between [0, 1] and a feedback string
def score_with_feedback(grader_prompt: str, y: Schema, m: InstanceMetadata) -> tuple[float, str]:
    test_case = {"task": m.gold["task"]}
    result_text = y["result"]

    grade = grade_by_model(grader_prompt, test_case, result_text)
    # grade_by_model returns a score between 1-10 -> normalize to [0, 1]
    raw_score = float(grade.get("score", 0))
    value = max(0.0, min(1.0, raw_score / 10.0))

    feedback_text = (
        f"Score: {raw_score}/10. "
        f"Strengths: {grade.get('strengths', [])}. "
        f"Weaknesses: {grade.get('weaknesses', [])}. "
        f"Reasoning: {grade.get('reasoning', '')}."
    )

    return value, feedback_text

# Executes control_flow C of a system S over an input x ∈ X_S
# One execution represent a specific LLM inference cost.
def rollout(s: System, x: Schema, m: InstanceMetadata, grader_prompt: str) -> list[float, str, list[Schema]]:
    y, traces = s(x)
    score, feedback = score_with_feedback(grader_prompt, y, m)
    return score, feedback, traces

def reflect_and_rewrite(old_prompt: str, traces: list[Schema], feedbacks: list[tuple[float, str]]) -> str:
    trace_blocks = ""
    for i, (t, (score, fb)) in enumerate(zip(traces, feedbacks)):
        new_trace = TRACE_BLOCK.format(
            n = i + 1,
            input = t.get("input"),
            output=t.get("output"),
            score=score,
            feedback=fb,
        )
        trace_blocks += (new_trace + "\n")
        compiled_prompt = REFLECTION_META_PROMPT.format(
            old_prompt=old_prompt,
            trace_blocks=trace_blocks,
        )

        messages = []
        add_user_message(messages, compiled_prompt)
        add_assistant_message(messages, "<new_instruction>")
        new = chat(messages, stop_sequences=["</new_instructions>"], temperature=1)

        return new.strip()