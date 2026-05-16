from typing import Any
import json
import random
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from evaluation import (
    chat,
    add_user_message,
    add_assistant_message,
    generate_dataset,
    run_prompt,
    grade_by_model,
    generate_run_uuid,
)
import numpy as np
from prompts import REFLECTION_META_PROMPT, TRACE_BLOCK
from spinner import Spinner

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
    if "gold" in m.gold:
        test_case["gold"] = m.gold["gold"]
    result_text = y["result"]

    grade = grade_by_model(grader_prompt, test_case, result_text)
    # grade_by_model returns a score between 1-5 -> normalize to [0, 1]
    raw_score = float(grade.get("score", 0))
    value = max(0.0, min(1.0, raw_score / 5.0))

    feedback_text = (
        f"Score: {raw_score}/5. "
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


def rollout_batch(
        system: System,
        examples: list[InstanceMetadata],
        grader_prompt: str,
        max_workers: int = 8,
) -> list[tuple[float, str, list[Schema]]]:
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(rollout, system, {"task": m.gold["task"]}, m, grader_prompt)
            for m in examples
        ]
        return [f.result() for f in futures]

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
    new = chat(messages, stop_sequences=["</new_instruction>"], temperature=1)

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
        k: sum(1 for ws in winners_per_instance if k in ws) for k in survivors
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
        result = run_prompt(prompt, x, temperature=1.0)
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

def _candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    return {"prompts": candidate.prompts, "parent_index": candidate.parent_index}


def _save_gepa_run(
        output_dir: Path,
        run_id: str,
        seed_prompt: str,
        best_prompt: str,
        pool: list[Candidate],
        scores: np.ndarray,
        rollout_used: int,
        rollout_budget: int,
        extra: dict[str, Any] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"gepa-{run_id}.json"
    payload: dict[str, Any] = {
        "run_id": run_id,
        "seed_prompt": seed_prompt,
        "best_prompt": best_prompt,
        "rollout_used": rollout_used,
        "rollout_budget": rollout_budget,
        "candidates": [_candidate_to_dict(candidate) for candidate in pool],
        "scores": scores.tolist(),
        "mean_scores": scores.mean(axis=1).tolist() if scores.size else [],
    }
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def run_gepa(
        seed_prompt: str, 
        dataset: list[dict], 
        grader_prompt: str, 
        rollout_budget: int = 50, 
        minibatch_size: int = 3,
        pareto_set_ratio: float = 0.4,
        output_folder_path: str | Path = ".output",
        run_id: str | None = None,
        save_progress: bool = True,
        spinner: Spinner | None = None,
        ) -> tuple[str, list[Candidate], np.ndarray]:

    if not dataset:
        raise ValueError("dataset must contain at least one test case")

    run_id = run_id or generate_run_uuid()
    output_dir = Path(output_folder_path)

    # split the dataset in two groups
    # feedback_set -> used to run the prompt and score the results at each rollout iteration
    # pareto_set -> used to score the result of the optimized prompt at every iteration
    # we split the set so that the optimizer nevers sees the pareto_set
    # -> does not over-fit the optimization on the dataset
    dataset = list(dataset)
    random.shuffle(dataset)
    split_index = max(1, int(len(dataset) * (1 - pareto_set_ratio)))
    if split_index >= len(dataset) and len(dataset) > 1:
        split_index = len(dataset) - 1
    feedback_set = [InstanceMetadata(gold=d) for d in dataset[:split_index]]
    pareto_source = dataset[split_index:] or dataset[:1]
    pareto_set = [InstanceMetadata(gold=d) for d in pareto_source]

    if save_progress:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"gepa-dataset-{run_id}.json", "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)

    base_module = Module(prompt=seed_prompt, in_schema={"task": str}, out_schema={"result": str}, run_inference=make_inference_fn())
    base_system = System(modules=[base_module], control_flow=single_module_control_flow, in_schema={"task": str}, out_schema={"result": str})

    pool: list[Candidate] = [Candidate(prompts=[seed_prompt], parent_index=None)]
    rollout_used = 0

    # seeding scoring
    if spinner:
        spinner.message = "Scoring seed prompt on Pareto set"
        spinner.start()
    try:
        initial = [
            score for score, _feedback, _traces in rollout_batch(base_system, pareto_set, grader_prompt)
        ]
    finally:
        if spinner:
            elapsed = spinner.stop()
            print(f"✓ Seed prompt scored in {elapsed:.2f}s")

    scores = np.array([initial])
    rollout_used = len(pareto_set)

    if save_progress:
        _save_gepa_run(
            output_dir=output_dir,
            run_id=run_id,
            seed_prompt=seed_prompt,
            best_prompt=seed_prompt,
            pool=pool,
            scores=scores,
            rollout_used=rollout_used,
            rollout_budget=rollout_budget,
            extra={"status": "seed_scored"},
        )

    # optimization loop
    iteration = 0
    if spinner:
        total_elapsed_time = 0
    try:
        while rollout_used < rollout_budget:
            if spinner:
                spinner.message = f"Running GEPA optimization | Iteration {iteration + 1}"
                spinner.start()

            parent_idx = select_candidate(scores)
            parent = pool[parent_idx]

            # NOTE: for our current implementation this is always 0
            j = select_module(iteration, len(base_system.modules))
            minibatch = feedback_set if len(feedback_set) <= minibatch_size else random.sample(feedback_set, minibatch_size)

            parent_system = build_system_from_candidate(parent, base_system)
            parent_results = rollout_batch(parent_system, minibatch, grader_prompt)
            parent_module_traces = [traces[j] for _score, _feedback, traces in parent_results]
            parent_feedbacks = [(score, feedback) for score, feedback, _traces in parent_results]
            rollout_used += len(minibatch)

            sigma_parent = sum(s for s, _ in parent_feedbacks) / len(parent_feedbacks)
            if rollout_used >= rollout_budget:
                break
            
            # according to reference paper's convention we DON'T count this API call in the budget
            new_prompt = reflect_and_rewrite(old_prompt=parent.prompts[j], traces=parent_module_traces, feedbacks=parent_feedbacks)

            child_prompts = list(parent.prompts)
            child_prompts[j] = new_prompt
            child = Candidate(prompts=child_prompts, parent_index=parent_idx)
            child_system = build_system_from_candidate(child, base_system)

            child_results = rollout_batch(child_system, minibatch, grader_prompt)
            child_feedbacks = [(score, feedback) for score, feedback, _traces in child_results]
            rollout_used += len(minibatch)
            sigma_child = sum(s for s, _ in child_feedbacks) / len(child_feedbacks)

            if sigma_child > sigma_parent and rollout_used < rollout_budget:
                child_pareto_scores = [
                    score for score, _feedback, _traces in rollout_batch(child_system, pareto_set, grader_prompt)
                ]
                rollout_used += len(pareto_set)
                pool.append(child)
                scores = np.vstack([scores, child_pareto_scores])

            iteration += 1
            if save_progress:
                best_index = int(scores.mean(axis=1).argmax())
                _save_gepa_run(
                    output_dir=output_dir,
                    run_id=run_id,
                    seed_prompt=seed_prompt,
                    best_prompt=pool[best_index].prompts[0],
                    pool=pool,
                    scores=scores,
                    rollout_used=rollout_used,
                    rollout_budget=rollout_budget,
                    extra={"status": "running", "iteration": iteration},
                )
            if spinner:
                elapsed = spinner.stop()
                total_elapsed_time += elapsed
                print(f"✓ Iteration {iteration} completed in {elapsed:.2f}s | Budget: {rollout_used}/{rollout_budget}")

    finally:
        if spinner:
            print(f"✓ GEPA optimization completed in {total_elapsed_time:.2f}s")

    best_index = int(scores.mean(axis=1).argmax())
    best_prompt = pool[best_index].prompts[0]
    if save_progress:
        path = _save_gepa_run(
            output_dir=output_dir,
            run_id=run_id,
            seed_prompt=seed_prompt,
            best_prompt=best_prompt,
            pool=pool,
            scores=scores,
            rollout_used=rollout_used,
            rollout_budget=rollout_budget,
            extra={"status": "completed", "iteration": iteration, "best_index": best_index},
        )
        print(f"✓ GEPA run saved to {path}")
    return best_prompt, pool, scores


if __name__ == "__main__":
    from evaluation import load_evaluation_prompts, validate

    prompts_folder = "example-prompts"
    output_dir = Path(".output")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = generate_run_uuid()

    target_prompt, dataset_prompt, grader_prompt = load_evaluation_prompts(
        prompts_folder=prompts_folder,
    )
    if not validate(target_prompt):
        raise SystemExit("[ERROR] target prompt missing {{task}} placeholder")

    spinner = Spinner("Generating dataset")
    spinner.start()
    try:
        dataset = generate_dataset(dataset_prompt)
    finally:
        elapsed = spinner.stop()
        print(f"✓ Dataset generated in {elapsed:.2f}s")

    dataset_path = output_dir / f"dataset-{run_id}.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    print(f"✓ Dataset saved to {dataset_path}")

    optimized, pool, scores = run_gepa(
        seed_prompt=target_prompt,
        dataset=dataset,
        grader_prompt=grader_prompt,
        rollout_budget=600,
        minibatch_size=4,
        output_folder_path=output_dir,
        run_id=run_id,
        save_progress=True,
        spinner=Spinner(),
    )

    optimized_path = output_dir / f"optimized-prompt-{run_id}.txt"
    with open(optimized_path, "w", encoding="utf-8") as f:
        f.write(optimized)
    print(f"✓ Optimized prompt saved to {optimized_path}")

    print("\nOptimized prompt:\n")
    print(optimized)
    print("\nCandidate scores:")
    for k, c in enumerate(pool):
        mean_s = float(scores[k].mean()) if len(scores[k]) else 0.0
        print(f"  Π_{k} (parent={c.parent_index}) mean_score={mean_s:.3f}")
