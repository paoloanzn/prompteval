from typing import Any
import random
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
import numpy as np
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

# example of scores
#          instance0  instance1  instance2
# Π_0  →   [0.3,      0.9,       0.1]
# Π_1  →   [0.95,     0.5,       0.7]
# Π_2  →   [0.7,      0.95,      0.2]
# Π = candidates
# instance = example from D_pareto

# scores[k][i] -> returns an index k
def select_candidate(scores: np.ndarray) -> int:
    # k, i
    n_candidates, n_instances = scores.shape
    
    best_per_instance = scores.max(axis=0) # winning scores per instance -> ([0.95, 0.95, 0.7])
    # indexes of winning candidates per instance -> [{1}, {2}, {1}]
    # len(winners_per_instance) == i and len(winners_per_instance[i]) <= k
    winners_per_instance: list[set[int]] = []

    for i in range(n_instances):
        col_i = scores[:, i] # all candidates scores for col i
        top_score = best_per_instance[i]
        is_winner_arr = top_score == col_i # np([False, True, True]) -> for each score in col_i tells if its the highest
        # NOTE: this works only with 1-DIM arrays
        winner_indexes = np.where(is_winner_arr)[0] # np([1, 2])
        winners_per_instance.append(set(winner_indexes.tolist()))

    # set of indexes [k] of all k elements(candidates) that won at least one instance i
    contenders: set[int] = set.union(*winners_per_instance)

    # compare two candidates k in scores[k][i] and return True if a dominates b
    # dominates -> highest score for every i OR same score for every i but at least higher in one i
    # a => [0.3,      0.9,       0.1]
    # b => [0.7,      0.95,      0.2]
    # b dominates a
    def dominates(a: int, b: int) -> bool:
        ge_all = all(scores[a][i] >= scores[b][i] for i in range(n_instances))
        gt_any = any(scores[a][i] >  scores[b][i] for i in range(n_instances))
        return ge_all and gt_any

    # drop ALL candidates that are dominated by at least 1 other candidate 
    # keep ONLY un-dominated candidates
    survivors: set[int] = {
        k for k in contenders if not any(dominates(other, k) for other in contenders if other != k)
    }

    # for every survivor we count how many instances they won
    fitness: dict[int, int] = {
        k: sum(1 for ws in winner_indexes if k in ws) for k in survivors
    }
    
    # pick only one candidate -> its fitness number becomes the probability weight of being chosen
    # the more instances i a candidate k won, the more probable is for it to be picked
    keys = list(fitness.keys())
    weights = list(fitness.values())
    return random.choices(keys, weights=weights, k=1)[0]