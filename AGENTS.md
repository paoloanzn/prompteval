# AGENTS.md

## Project Overview

- Stack: Python CLI for a mini GEPA prompt optimizer using Anthropic, OpenRouter, OpenAI, NumPy, dotenv, and json_repair.
- Versions: Python 3.14.4 locally; dependencies are `anthropic>=0.100.0`, `numpy>=2.0`, `python-dotenv>=1.0`, `json_repair>=0.59.9`, `openrouter==0.9.1`, and `openai==2.37.0`.
- Package manager: `pip` with `requirements.txt`; no lockfile, `pyproject.toml`, or package script runner.

## Commands

| Purpose | Command |
| --- | --- |
| Create virtualenv | `python3 -m venv .venv` |
| Activate virtualenv | `source .venv/bin/activate` |
| Install dependencies | `python -m pip install -r requirements.txt` |
| Generate dataset with explicit defaults | `.venv/bin/python cli.py generate --prompts example-prompts --output .output` |
| Evaluate and generate dataset if omitted | `.venv/bin/python cli.py evaluate --prompts example-prompts --output .output` |
| Evaluate with an existing dataset | `.venv/bin/python cli.py evaluate --prompts example-prompts --output .output --dataset .output/dataset-abc123.json` |
| Optimize with explicit defaults | `.venv/bin/python cli.py optimize --prompts example-prompts --output .output --budget 120 --minibatch 3 --pareto-ratio 0.4` |
| Optimize with dataset and custom settings | `.venv/bin/python cli.py optimize --prompts example-prompts --output .output --dataset .output/dataset-abc123.json --budget 300 --minibatch 6 --pareto-ratio 0.4` |
| Start prompt-creation REPL | `.venv/bin/python repl.py` |
| Run no-network syntax check | `.venv/bin/python -m compileall -q cli.py client.py evaluation.py gepa.py prompts.py repl.py spinner.py` |

## Code Style

Imports:

```python
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
```

Typed helpers:

```python
def _load_or_generate_dataset(
    dataset_path: str | None,
    dataset_prompt: str | None,
    output_dir: Path,
    run_id: str,
) -> tuple[list[dict], Path]:
    if dataset_path:
        path = Path(dataset_path)
        if not path.is_file():
            raise SystemExit(f"[ERROR] dataset file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"Loaded dataset from {path} ({len(dataset)} test cases)")
        return dataset, path
```

File writes:

```python
with open(eval_path, "w", encoding="utf-8") as f:
    json.dump(evaluation, f, indent=2)
```

CLI validation:

```python
if not validate(target_prompt):
    raise SystemExit("[ERROR] target prompt missing {{task}} placeholder")
```

Comment style:

```python
# grade_by_model returns a score between 1-5 -> normalize to [0, 1]
raw_score = float(grade.get("score", 0))
value = max(0.0, min(1.0, raw_score / 5.0))
```

Use short `#` comments for non-obvious logic or constraints. Do not add docstrings just to restate function names, parameters, or return types.

## Testing

- There are no tests in this repo for now, and there is no `tests/` directory or pytest configuration.
- No-network check before commits: `.venv/bin/python -m compileall -q cli.py client.py evaluation.py gepa.py prompts.py repl.py spinner.py`.
- When tests are added, place them in `tests/` as `test_*.py` files.
- Write tests for deterministic helpers, JSON parsing/repair behavior, prompt path resolution, schema/candidate selection, CLI argument behavior, and regressions that do not require live model calls.
- Avoid tests that hit Anthropic, OpenRouter, or OpenAI unless the user explicitly asks for live API coverage.

## Git/PR Workflow

- Recent human commit subjects: `(refactor) simplified make_client and removed "bearer" auth_mode literal type`, `(docs) update README`, `(chore) update requirements to include openrouter and openai packages`, `(feat) update chat function to support teacher-client model for dataset generation and grading`.
- Human commit format: `(type) imperative summary`.
- AI-created commit format: `(type) ([model], reviewed T|F, tested T|F) imperative summary`.
- AI-created commit example: `(docs) (openai/gpt-5.5, reviewed F, tested T) add agent guide`.
- PR title format should match the primary commit subject. For AI-created PRs, keep the same model/review/test prefix in the title.
- Required pre-commit check: `git status --short`.
- Required pre-commit check: `git diff --check`.
- Required pre-commit check: `.venv/bin/python -m py_compile *.py`.
- Only mark `tested T` after the relevant checks have run successfully. Use `tested F` when checks were not run, failed, or require unavailable API credentials.

## Boundaries

- Never edit, print, copy, commit, or infer values from `.env`, API keys, OAuth tokens, or provider credentials.
- Never touch dependency/vendor/runtime directories: `.venv/`, `venv/`, `__pycache__/`, or any future `vendor/` directory.
- Never modify generated run artifacts in `.output/` unless the user explicitly asks.
- Never edit `.git/` internals, hooks, refs, logs, or local git config.
- Never change production/provider configuration without explicit approval: provider base URLs, auth modes, environment variable names, default model names, and live API behavior in `client.py`.
- Never commit local OS/editor artifacts such as `.DS_Store`.

## Best Coding Practices

- Never do large refactors at once unless the user approves.
- Prefer small incremental changes with small surface area impact.
- Write simple clear code without creating duplicate code throughout the repo.
