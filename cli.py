import argparse
import json
from pathlib import Path

from evaluation import (
    load_evaluation_prompts,
    validate,
    generate_dataset,
    run_eval,
    generate_run_uuid,
)
from gepa import run_gepa
from spinner import Spinner


def cmd_generate(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, dataset_prompt, _ = load_evaluation_prompts(prompts_folder=args.prompts)

    spinner = Spinner("Generating dataset")
    spinner.start()
    try:
        dataset = generate_dataset(dataset_prompt)
    finally:
        elapsed = spinner.stop()
        print(f"✓ Dataset generated in {elapsed:.2f}s")

    run_id = generate_run_uuid()
    dataset_path = output_dir / f"dataset-{run_id}.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"✓ Dataset saved to {dataset_path}")
    print(f"  {len(dataset)} test cases")


def cmd_evaluate(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = generate_run_uuid()

    target_prompt, dataset_prompt, grader_prompt = load_evaluation_prompts(
        prompts_folder=args.prompts,
    )
    if not validate(target_prompt):
        raise SystemExit("[ERROR] target prompt missing {{task}} placeholder")

    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.is_file():
            raise SystemExit(f"[ERROR] dataset file not found: {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"✓ Loaded dataset from {dataset_path} ({len(dataset)} test cases)")
    else:
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

    spinner = Spinner("Running evaluation")
    spinner.start()
    try:
        evaluation = run_eval(target_prompt, grader_prompt, dataset)
    finally:
        elapsed = spinner.stop()
        print(f"✓ Evaluation completed in {elapsed:.2f}s")

    eval_path = output_dir / f"evaluation-{run_id}.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2)
    print(f"✓ Evaluation saved to {eval_path}")

    # summary
    total = len(evaluation)
    errors = sum(1 for e in evaluation if e.get("error"))
    scored = [e for e in evaluation if not e.get("error") and "score" in e]
    mean_score = sum(e["score"] for e in scored) / len(scored) if scored else 0

    print(f"\n  Total cases: {total}")
    print(f"  Scored:      {len(scored)}")
    print(f"  Errors:      {errors}")
    print(f"  Mean score:  {mean_score:.2f}/5")


def cmd_optimize(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = generate_run_uuid()

    target_prompt, dataset_prompt, grader_prompt = load_evaluation_prompts(
        prompts_folder=args.prompts,
    )
    if not validate(target_prompt):
        raise SystemExit("[ERROR] target prompt missing {{task}} placeholder")

    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.is_file():
            raise SystemExit(f"[ERROR] dataset file not found: {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"✓ Loaded dataset from {dataset_path} ({len(dataset)} test cases)")
    else:
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
        rollout_budget=args.budget,
        minibatch_size=args.minibatch,
        pareto_set_ratio=args.pareto_ratio,
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


def main():
    parser = argparse.ArgumentParser(
        prog="prompteval",
        description="GEPA prompt optimizer CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate a dataset from a dataset prompt")
    gen_parser.add_argument("--prompts", default="example-prompts", help="Prompt folder (default: example-prompts)")
    gen_parser.add_argument("--output", default=".output", help="Output folder (default: .output)")
    gen_parser.set_defaults(func=cmd_generate)

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a target prompt against a dataset")
    eval_parser.add_argument("--prompts", default="example-prompts", help="Prompt folder (default: example-prompts)")
    eval_parser.add_argument("--output", default=".output", help="Output folder (default: .output)")
    eval_parser.add_argument("--dataset", help="Path to existing dataset JSON (generates one if omitted)")
    eval_parser.set_defaults(func=cmd_evaluate)

    # optimize
    opt_parser = subparsers.add_parser("optimize", help="Run GEPA optimization on a target prompt")
    opt_parser.add_argument("--prompts", default="example-prompts", help="Prompt folder (default: example-prompts)")
    opt_parser.add_argument("--output", default=".output", help="Output folder (default: .output)")
    opt_parser.add_argument("--dataset", help="Path to existing dataset JSON (generates one if omitted)")
    opt_parser.add_argument("--budget", type=int, default=120, help="Rollout budget (default: 120)")
    opt_parser.add_argument("--minibatch", type=int, default=3, help="Minibatch size (default: 3)")
    opt_parser.add_argument("--pareto-ratio", type=float, default=0.4, help="Pareto set ratio (default: 0.4)")
    opt_parser.set_defaults(func=cmd_optimize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
