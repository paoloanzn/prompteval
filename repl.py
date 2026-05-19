#!/usr/bin/env python3
import glob as globlib
import json
import os
import re
import subprocess
from pathlib import Path

from client import chat, teacher_client, teacher_model


RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
BLUE, CYAN, GREEN, RED = "\033[34m", "\033[36m", "\033[32m", "\033[31m"


def read(args):
    lines = open(args["path"], encoding="utf-8").readlines()
    offset = args.get("offset", 0)
    limit = args.get("limit", len(lines))
    selected = lines[offset : offset + limit]
    return "".join(f"{offset + idx + 1:4}| {line}" for idx, line in enumerate(selected))


def write(args):
    path = Path(args["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(args["content"])
    return "ok"


def edit(args):
    path = args["path"]
    text = open(path, encoding="utf-8").read()
    old, new = args["old"], args["new"]
    if old not in text:
        return "error: old_string not found"
    count = text.count(old)
    if not args.get("all") and count > 1:
        return f"error: old_string appears {count} times, must be unique (use all=true)"
    text = text.replace(old, new) if args.get("all") else text.replace(old, new, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return "ok"


def glob(args):
    pattern = (args.get("path", ".") + "/" + args["pat"]).replace("//", "/")
    files = globlib.glob(pattern, recursive=True)
    files = sorted(files, key=lambda f: os.path.getmtime(f) if os.path.isfile(f) else 0, reverse=True)
    return "\n".join(files) or "none"


def grep(args):
    pattern = re.compile(args["pat"])
    hits = []
    for filepath in globlib.glob(args.get("path", ".") + "/**", recursive=True):
        try:
            for line_num, line in enumerate(open(filepath, encoding="utf-8"), 1):
                if pattern.search(line):
                    hits.append(f"{filepath}:{line_num}:{line.rstrip()}")
        except Exception:
            pass
    return "\n".join(hits[:50]) or "none"


def bash(args):
    proc = subprocess.Popen(args["cmd"], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = []
    try:
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                print(f"  {DIM}│ {line.rstrip()}{RESET}", flush=True)
                output.append(line)
        proc.wait(timeout=args.get("timeout", 120))
    except subprocess.TimeoutExpired:
        proc.kill()
        output.append("\n(timed out)")
    return "".join(output).strip() or "(empty)"


TOOLS = {
    "read": ("Read a file with line numbers", {"path": "string", "offset": "number?", "limit": "number?"}, read),
    "write": ("Write content to a file, creating parent directories", {"path": "string", "content": "string"}, write),
    "edit": ("Replace old with new in a file", {"path": "string", "old": "string", "new": "string", "all": "boolean?"}, edit),
    "glob": ("Find files by glob pattern", {"pat": "string", "path": "string?"}, glob),
    "grep": ("Search files for a regex pattern", {"pat": "string", "path": "string?"}, grep),
    "bash": ("Run a shell command", {"cmd": "string", "timeout": "number?"}, bash),
}


def run_tool(name, args):
    try:
        return TOOLS[name][2](args)
    except Exception as err:
        return f"error: {err}"


def make_schema():
    result = []
    for name, (description, params, _fn) in TOOLS.items():
        properties, required = {}, []
        for param_name, param_type in params.items():
            optional = param_type.endswith("?")
            base_type = param_type.rstrip("?")
            properties[param_name] = {"type": "integer" if base_type == "number" else base_type}
            if not optional:
                required.append(param_name)
        result.append({"name": name, "description": description, "input_schema": {"type": "object", "properties": properties, "required": required}})
    return result


def blocks(response):
    if hasattr(response, "content"):
        return [block.model_dump() for block in response.content]
    message = response.choices[0].message
    result = []
    if getattr(message, "content", None):
        result.append({"type": "text", "text": message.content})
    for call in getattr(message, "tool_calls", None) or []:
        result.append({"type": "tool_use", "id": call.id, "name": call.function.name, "input": json.loads(call.function.arguments or "{}")})
    return result


def separator():
    width = min(os.get_terminal_size().columns, 80) if os.isatty(1) else 80
    return f"{DIM}{'-' * width}{RESET}"


def render_markdown(text):
    return re.sub(r"\*\*(.+?)\*\*", f"{BOLD}\\1{RESET}", text)


def main():
    print(f"{BOLD}minigepa repl{RESET} | {DIM}{teacher_model} | {os.getcwd()}{RESET}\n")
    messages = []
    system = f"""Concise minigepa prompt assistant. cwd: {os.getcwd()}

Help the user create and iterate on a prompt folder containing exactly:
- target_prompt.txt: prompt being optimized, must include {{{{task}}}}
- dataset_prompt.txt: asks for JSON dataset list; each item needs task and may include gold
- grader_prompt.txt: grades {{{{result}}}} for {{{{task}}}}, may use {{{{gold}}}}, returns JSON with score 1-5

When creating or editing prompts, first inspect example-prompts/ and follow the same structure used by its three prompt files.

Use tools to inspect, write, and edit files. Save prompt files when the user asks.
To run the project use bash, for example:
- .venv/bin/python cli.py generate --prompts <folder>
- .venv/bin/python cli.py evaluate --prompts <folder> [--dataset path]
- .venv/bin/python cli.py optimize --prompts <folder> [--dataset path] [--budget 120]
"""

    while True:
        try:
            print(separator())
            user_input = input(f"{BOLD}{BLUE}❯{RESET} ").strip()
            print(separator())
            if not user_input:
                continue
            if user_input in ("/q", "exit", "quit"):
                break
            if user_input == "/c":
                messages = []
                print(f"{GREEN}⏺ Cleared conversation{RESET}")
                continue

            messages.append({"role": "user", "content": user_input})
            while True:
                response = chat(messages, system=system, tools=make_schema(), raw=True, _client=teacher_client, _model=teacher_model)
                content_blocks = blocks(response)
                tool_results = []

                for block in content_blocks:
                    if block["type"] == "text":
                        print(f"\n{CYAN}⏺{RESET} {render_markdown(block['text'])}")
                    if block["type"] == "tool_use":
                        name, args = block["name"], block["input"]
                        preview = str(next(iter(args.values()), ""))[:50]
                        print(f"\n{GREEN}⏺ {name}{RESET}({DIM}{preview}{RESET})")
                        result = run_tool(name, args)
                        lines = result.split("\n")
                        suffix = f" ... +{len(lines) - 1} lines" if len(lines) > 1 else ""
                        print(f"  {DIM}⎿  {lines[0][:60]}{suffix}{RESET}")
                        tool_results.append({"type": "tool_result", "tool_use_id": block["id"], "content": result})

                messages.append({"role": "assistant", "content": content_blocks})
                if not tool_results:
                    break
                messages.append({"role": "user", "content": tool_results})
            print()
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as err:
            print(f"{RED}⏺ Error: {err}{RESET}")


if __name__ == "__main__":
    main()
