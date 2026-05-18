from __future__ import annotations
from dotenv import load_dotenv
from dataclasses import dataclass
from anthropic import Anthropic
from openai import OpenAI
from openrouter import OpenRouter
import os
from enum import StrEnum
from typing import Any, Literal, Mapping, Union

load_dotenv()

class Provider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    OPENAI = "openai"

auth_mode = Literal["x-api-key", "oauth"]
api_contract = Literal["anthropic", "openai", "openrouter"]

@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    auth_mode: auth_mode
    env_key: str
    api_contract: api_contract

PROVIDERS: dict[Provider, ProviderConfig] = {
    Provider.ANTHROPIC: ProviderConfig(
        base_url="https://api.anthropic.com",
        auth_mode="oauth",
        env_key="ANTHROPIC_OAUTH_TOKEN",
        api_contract="anthropic",
    ),
    Provider.OPENROUTER: ProviderConfig(
        base_url="https://openrouter.ai/api/v1",
        auth_mode="x-api-key",
        env_key="OPENROUTER_API_KEY",
        api_contract="openrouter",
    ),
    Provider.OPENAI: ProviderConfig(
        base_url="https://api.openai.com/v1",
        auth_mode="x-api-key",
        env_key="OPENAI_API_KEY",
        api_contract="openai",
    ),
}

Client = Union[Anthropic, OpenAI, OpenRouter]

def make_client(
    provider: Provider | str = Provider.ANTHROPIC,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    default_headers: Mapping[str, str] | None = None,
    default_query: Mapping[str, object] | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> Client:
    provider = Provider(provider)
    config = PROVIDERS[provider]

    key = api_key or os.environ.get(config.env_key)
    if not key:
        raise ValueError(f"Missing {config.env_key}")

    if config.api_contract == "openrouter":
        kwargs: dict[str, Any] = {"api_key": key}
        if timeout is not None:
            kwargs["timeout_ms"] = int(timeout * 1000)
        if max_retries is not None:
            from openrouter.utils.retries import RetryConfig
            kwargs["retry_config"] = RetryConfig(max_retries=max_retries)
        return OpenRouter(**kwargs)

    kwargs: dict[str, Any] = {
        "base_url": base_url or config.base_url,
        "default_headers": dict(default_headers or {}) or None,
        "default_query": dict(default_query or {}) or None,
    }
    kwargs["auth_token" if config.auth_mode == "oauth" else "api_key"] = key
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return Anthropic(**kwargs) if config.api_contract == "anthropic" else OpenAI(**kwargs)

# client init
# we use a student(client)-teacher(teacher_client) model -> scoring and dataset gen require smarter models

# used by: run_prompt(), reflect_and_rewrite()
client = (
    make_client(Provider.OPENROUTER)
    if os.environ.get(PROVIDERS[Provider.OPENROUTER].env_key)
    else None
)
model = os.environ.get("STUDENT", "nvidia/nemotron-3-nano-30b-a3b")

# used by: generate_dataset(), grade_by_model()
teacher_client = make_client(Provider.ANTHROPIC)
teacher_model = os.environ.get("TEACHER", "claude-haiku-4-5")

if not model or not teacher_model:
    raise ValueError(f"{model if not model else teacher_model} model not defined.")

def add_user_message(messages, text):
    user_message = {"role": "user", "content": text}
    messages.append(user_message)


def add_assistant_message(messages, text):
    assistant_message = {"role": "assistant", "content": text}
    messages.append(assistant_message)


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(item.text for item in content if hasattr(item, 'text') and item.text)
    return ""

# tools calls and tool results in message history are handled differently in OpenAI and Openrouter -> needs conversion
def _openai_tools(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object"}),
            },
        }
        for tool in tools
    ]

def _openai_messages(messages: list[dict], system: str | None = None) -> list[dict]:
    import json

    def convert(message: dict) -> list[dict]:
        role = message.get("role")
        content = message.get("content")

        if not isinstance(content, list):
            return [message]

        text = "".join(
            block.get("text", "")
            for block in content
            if block.get("type") == "text"
        )

        if role == "assistant":
            tool_calls = [
                {
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block.get("input", {})),
                        },
                }
                for block in content
                if block.get("type") == "tool_use"
            ]

            return [{
                "role": "assistant",
                "content": text or (None if tool_calls else ""),
                **({"tool_calls": tool_calls} if tool_calls else {}),
            }]

        if role == "user":
            return [
                *[
                    {
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": block.get("content", ""),
                    }
                    for block in content
                    if block.get("type") == "tool_result"
                ],
                *([{"role": "user", "content": text}] if text else []),
            ]

        return [message]

    return [
        *([{"role": "system", "content": system}] if system else []),
        *(item for message in messages for item in convert(message)),
    ]


def chat(
        messages,
        system=None,
        temperature=1.0,
        stop_sequences=[],
        max_tokens=8000,
        tools: list[dict] | None = None,
        raw: bool = False,
        *,
        _client: Client | None = client,
        _model: str = model,
):
    if _client is None:
        raise ValueError("Missing OPENROUTER_API_KEY for student client")

    if isinstance(_client, Anthropic):
        params = {
            "model": _model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
            "stop_sequences": stop_sequences,
            "thinking": {"type": "disabled"},
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = tools
        message = _client.messages.create(**params)
        if raw:
            return message
        return "".join(block.text for block in message.content if block.type == "text")

    if isinstance(_client, OpenRouter):
        from openrouter.components.chatrequest import Reasoning
        params: dict[str, Any] = {
            "model": _model,
            "max_tokens": max_tokens,
            "messages": _openai_messages(messages, system),
            "temperature": temperature,
            "reasoning": Reasoning(effort="none"),
        }
        if stop_sequences:
            params["stop"] = stop_sequences
        if tools:
            params["tools"] = _openai_tools(tools)
        response = _client.chat.send(**params)
        if raw:
            return response
        return _extract_text(response.choices[0].message.content)

    # OpenAI
    params = {
        "model": _model,
        "max_tokens": max_tokens,
        "messages": _openai_messages(messages, system),
        "temperature": temperature,
    }
    if stop_sequences:
        params["stop"] = stop_sequences
    if tools:
        params["tools"] = _openai_tools(tools)
    completion = _client.chat.completions.create(**params)
    if raw:
        return completion
    return _extract_text(completion.choices[0].message.content)
