from typing import Any
from dataclasses import dataclass
from collections.abc import Callable

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

# S = (M, C, X, Y)
# S(x -> instance of Schema) returns y instance of Schema
# x must match S.in_schema and y must match S.out_schema
class System:
    modules: list[Module]
    control_flow: Callable[[Schema], Schema]
    in_schema: Schema
    out_schema: Schema

    def __call__(self, x: Schema) -> list[Schema, list[Schema]]:
        if not matches_schema(x, self.in_schema):
            raise ValueError("Input does not match system input schema")

        y, traces = self.control_flow(x)

        if not matches_schema(y, self.out_schema):
            raise ValueError("Output does not match system output schema")

        return y, traces


# Mi = (Pi, Wi, Xi, Yi) 
# Wi, model weights -> not relevant for GEPA
class Module:
    prompt: str # Pi -> mutated
    in_schema: Schema
    out_schema: Schema
    run_inference: Callable[[Schema], Schema]
    
    def __call__(self, x: Schema) -> Schema:
        if not matches_schema(x, self.in_schema):
            raise ValueError("Input does not match module input schema")

        y, traces = self.run_inference(x)

        if not matches_schema(y, self.out_schema):
            raise ValueError("Output does not match module output schema")

        return y, traces

# Bundle of prompts of every module Mi in the system S
class Candidate:
    prompts: list[str]


# what the system does not see that is needed to score the output y in shape Y of it
class InstanceMetadata:
    pass

# Executes control_flow C of a system S over an input x ∈ X_S
# One execution represent a specific LLM inference cost.
def rollout(s: System, x: Schema, m: InstanceMetadata) -> list[float, str, list[Any]]:
    pass