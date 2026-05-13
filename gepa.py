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
@dataclass
class Candidate:
    prompts: list[str]
    parent_index: int | None = None

# what the system does not see that is needed to score the output y in shape Y of it
@dataclass
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

# round robin module selection
# NOTE: in this one-module system implementation this will always return the index of the only module present -> 0
def select_module(iteration: int, n_modules: int) -> int:
    return iteration % n_modules

# factory function
# NOTE: currently supports only -> {system_prompt, task} module type
def make_inference_fn() -> Callable[[Schema, str], tuple[Schema, Schema]]:
    def run_inference(x: Schema, prompt: str) -> tuple[Schema, Schema]:
        result = run_prompt(prompt, x, 1.0)
        trace = {"input": x, "output": result, "prompt": prompt}
        return {"result": result}, trace
    return run_inference

def single_module_control_flow(modules: list[Module], x: Schema) -> tuple[Schema, list[Schema]]:
    y, trace = modules[0](x)
    return y, [trace]

# NOTE: in this implementation j == 0
def build_system_from_candidate(candidate: Candidate, base_system: System) -> System:
    new_modules = [Module(
        prompt=candidate.prompts[j],
        in_schema=base_system.modules[j].in_schema,
        out_schema=base_system.modules[j].out_schema,
        run_inference=base_system.modules[j].run_inference,
    ) for j in range(len(base_system.modules))]

    return System(
        new_modules, 
        control_flow=base_system.control_flow, 
        in_schema=base_system.in_schema, 
        out_schema=base_system.out_schema
        )

def run_gepa(
        seed_prompt: str, 
        dataset: list[str], 
        grader_prompt: str, 
        rollout_budget: int = 50, 
        minibatch_size: int = 3,
        pareto_set_ratio: float = 0.4
        ) -> tuple[str, list[Candidate], np.ndarray]:

    # split the dataset in two groups
    # feedback_set -> used to run the prompt and score the results at each rollout iteration
    # pareto_set -> used to score the result of the optimized prompt at every iteration
    # we split the set so that the optimizer nevers sees the pareto_set
    # -> does not over-fit the optimization on the dataset
    random.shuffle(dataset)
    split_index = max(1, int(len(dataset) * (1 - pareto_set_ratio)))
    feedback_set = [InstanceMetadata(gold=d) for d in dataset[:split_index]]
    pareto_set = [InstanceMetadata(gold=d) for d in dataset[split_index:]]

    base_module = Module(prompt=seed_prompt, in_schema={"task", str}, out_schema={"result", str}, run_inference=make_inference_fn())
    base_system = System(modules=[base_module], control_flow=single_module_control_flow, in_schema={"task": str}, out_schema={"result": str})

    pool: list[Candidate] = [Candidate(prompts=[seed_prompt], parent_index=None)]
    rollout_used = 0

    # seeding scoring
    initial = []
    for m in pareto_set:
        score, _feedback, _traces = rollout(base_system, {"task": m.gold["task"]}, m, grader_prompt)
        initial.append(score)
    scores = np.array([initial])
    rollout_used = len(pareto_set)

    # optimization loop
    iteration = 0
    while rollout_used < rollout_budget:
        parent_idx = select_candidate(scores)
        parent = pool[parent_idx]

        # NOTE: for our current implementation this is always 0
        j = select_module(iteration, len(base_system.modules))
        
        # pick minibatch_size_N random examples from d_feedback
        # OR
        # pick all d_feedback if there are less then minibatch_size
        minibatch = []
        if len(feedback_set) < minibatch_size:
            minibatch = feedback_set
        else:
            minibatch = random.sample(feedback_set, minibatch_size)

        parent_system = build_system_from_candidate(parent, base_system)
        parent_module_traces: list[Schema] = []
        parent_feedbacks: list[tuple[float, str]] = []
        for m in minibatch:
            score, feedback, traces = rollout(parent_system, {"task": m.gold["task"]}, m, grader_prompt)
            parent_module_traces.append[traces[j]]
            parent_feedbacks.append[(score, feedback)]
        rollout_used += len(minibatch)

        # compute avg score in the minibatch
        sigma_parent = sum(s for s, _ in parent_feedbacks) / len(parent_feedbacks)

        # stop when budget is exhausted 
        if rollout_used >= rollout_budget:
            break
        
        # according to reference paper's convention we DON'T count this API call in the budget
        new_prompt = reflect_and_rewrite(old_prompt=parent.prompts[j], traces=parent_module_traces, feedbacks=parent_feedbacks)

        child_prompts = list(parent.prompts)
        child_prompts[j] = new_prompt
        # parent_index tell us from which prompt's module the child came from
        child = Candidate(prompts=child_prompts, parent_index=parent_idx)

        child_system = build_system_from_candidate(child, base_system)
        child_feedbacks: list[tuple[float, str]] = []
        for m in minibatch:
            score, feedback, _ = rollout(
                child_system, {"task": m.gold["task"]}, m, grader_prompt
            )
            child_feedbacks.append((score, feedback))
        rollout_used += len(minibatch)
        sigma_child = sum(s for s, _ in child_feedbacks) / len(child_feedbacks)

        # rollout budget is spent ONLY if the child scored better then the parent
        if sigma_child > sigma_parent and rollout_used < rollout_budget:
            child_pareto_scores: list[float] = []
            for m in pareto_set:
                score, _, _ = rollout(
                    child_system, {"task": m.gold["task"]}, m, grader_prompt
                )
            child_pareto_scores.append(score)
            rollout_used += len(pareto_set)
            pool.append[child]
            scores = np.vstack([scores, child_pareto_scores])

            iteration += 1
        
        best_index = int(scores.mean(axis=1).argmax())
        return pool[best_index].prompts[0], pool, scores