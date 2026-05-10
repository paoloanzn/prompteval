from dotenv import load_dotenv
from anthropic import Anthropic
import os
import json
import sys
from pathlib import Path
from uuid import uuid4
from spinner import Spinner

load_dotenv()

client = Anthropic(auth_token=os.environ["ANTHROPIC_OAUTH_TOKEN"])
model = os.environ.get("MODEL", "claude-haiku-4-5")

# anthropic helpers
def add_user_message(messages, text):
    user_message = {"role": "user", "content": text}
    messages.append(user_message)


def add_assistant_message(messages, text):
    assistant_message = {"role": "assistant", "content": text}
    messages.append(assistant_message)

def chat(messages, system=None, temperature=1.0, stop_sequences=[]):
    params = {
        "model": model,
        "max_tokens": 8000,
        "messages": messages,
        "temperature": temperature,
        "stop_sequences": stop_sequences,
    }

    if system:
        params["system"] = system

    message = client.messages.create(**params)
    return message.content[0].text

# prompt helpers

# target prompt -> prompt to evaluate
# dataset prompt -> prompt that generates the dataset
# grader prompt -> prompt that generates the grade

# replace placeholder in this form {{ <var> }} and returned the resulting prompt
def compile_prompt_template(prompt: str, vars: dict) -> str:
    res = prompt
    for key in vars.keys():
        placeholder = "{{" + key + "}}"  # {{ key }}
        res = res.replace(placeholder, vars[key])
    return res

# load prompt from a file a file on disk
def load_prompt(file_path: Path | str) -> str:
    prompt = None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            prompt = f.read()
        return prompt
    except Exception:
        return prompt

# checks that a target prompt contains `{{task}}` -> required for test eval
def is_valid_target_prompt(prompt: str) -> bool:
    res = True if "{{"+ "task" + "}}" in prompt else False
    return res 

# resolve prompt file paths from a folder, looking for .txt then .md files
# returns (target_prompt_path, dataset_prompt_path, grader_prompt_path)
def resolve_prompt_paths(folder: str | Path) -> tuple[Path, Path, Path]:
    folder = Path(folder)
    prompt_names = ["target_prompt", "dataset_prompt", "grader_prompt"]
    paths = []
    for name in prompt_names:
        found = None
        for ext in [".txt", ".md"]:
            candidate = folder / f"{name}{ext}"
            if candidate.is_file():
                found = candidate
                break
        if found is None:
            raise FileNotFoundError(f"Could not find {name}.txt or {name}.md in {folder}")
        paths.append(found)
    return tuple(paths)

class EvaluationSpace():
    def __init__(
            self,
            prompts_folder: str | Path = None,
            target_prompt_path: str | Path = None,
            dataset_prompt_path: str | Path = None,
            grader_prompt_path: str | Path = None,
    ):
        self.target_prompt = None
        self.dataset_prompt = None
        self.grader_prompt = None

        if prompts_folder:
            target_prompt_path, dataset_prompt_path, grader_prompt_path = \
                resolve_prompt_paths(prompts_folder)

        if not all([target_prompt_path, dataset_prompt_path, grader_prompt_path]):
            raise ValueError(
                "Must provide either prompts_folder or all three of "
                "target_prompt_path, dataset_prompt_path, grader_prompt_path"
            )

        with open(target_prompt_path, "r", encoding="utf-8") as f:
            self.target_prompt = f.read()
        with open(dataset_prompt_path, "r", encoding="utf-8") as f:
            self.dataset_prompt = f.read()
        with open(grader_prompt_path, "r", encoding="utf-8") as f:
            self.grader_prompt = f.read()

    # run prompt validation functions
    def _validate(self) -> bool:
        is_valid = is_valid_target_prompt(self.target_prompt)
        return is_valid
 
    def generate_dataset(self) -> list[dict]:
        messages = []
        add_user_message(messages, self.dataset_prompt)
        add_assistant_message(messages, "```json")
        text = chat(messages, stop_sequences=["```"])
        return json.loads(text)

    # runs the prompt to evaluate against a test case from the generated dataset 
    def run_prompt(self, test_case: dict) -> str:
        messages = []
        add_user_message(messages, compile_prompt_template(self.target_prompt, { "task": test_case["task"] }))
        output = chat(messages)
        return output

    # run the grading prompt with test case and test case result
    def grade_by_model(self, test_case: dict, result: str) -> dict:
        messages = []
        add_user_message(messages, compile_prompt_template(self.grader_prompt, { "task": test_case["task"], "result": result}))
        add_assistant_message(messages, "```json")
        eval_text = chat(messages, stop_sequences=["```"], temperature=0) # frozen inference -> grader should be deterministic-ish

        return json.loads(eval_text)

    # run a test case with run_prompt + grade_by_model -> return both result and evaluation
    def run_test_case(self, test_case: dict) -> dict:
        try:
            result = self.run_prompt(test_case)
            grade = self.grade_by_model(test_case, result)
            return {"result": result, **grade, "error": None}
        except Exception as err:
            return {"result": None, "error": err}
    
    # run run_test_case for all test cases in a dataset
    def run_eval(self, dataset: dict) -> list:
        dataset_eval = []
        for test_case in dataset:
            dataset_eval.append(self.run_test_case(test_case))
        return dataset_eval

    def _generate_run_uuid(self) -> str:
        return uuid4().hex[:8]

    def run(self, output_folder_path: str = ".output") -> None:
        if not self._validate():
            print("[ERROR] Target prompt is not valid.")
            sys.exit(1)

        run_id = self._generate_run_uuid()
        output_dir = Path(output_folder_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # generate dataset
        spinner = Spinner("Generating dataset")
        spinner.start()
        dataset = self.generate_dataset()
        elapsed = spinner.stop()
        print(f"\u2713 Dataset generated in {elapsed:.2f}s")

        # save dataset to disk
        with open(output_dir / f"dataset-{run_id}.json", "w") as f:
            json.dump(dataset, f, indent=2)

        # run evaluation
        spinner = Spinner("Running evaluation")
        spinner.start()
        evaluation = self.run_eval(dataset)
        elapsed = spinner.stop()
        print(f"\u2713 Evaluation completed in {elapsed:.2f}s")

        # save evaluation to disk
        with open(output_dir / f"evaluation-{run_id}.json", "w") as f:
            json.dump(evaluation, f, indent=2)
        
if __name__ == "__main__":
    eval_space = EvaluationSpace("example-prompts")
    eval_space.run()