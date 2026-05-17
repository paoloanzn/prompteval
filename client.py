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

auth_mode = Literal["x-api-key", "bearer", "oauth"]
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
        auth_mode="bearer",
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

    if config.api_contract == "anthropic":
        base_url = base_url or config.base_url
        headers: dict[str, str] = dict(default_headers or {})
        query: dict[str, object] = dict(default_query or {})
        kwargs: dict[str, Any] = {
            "base_url": base_url,
            "default_headers": headers or None,
            "default_query": query or None,
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
        if config.auth_mode == "x-api-key":
            kwargs["api_key"] = key
        else:
            kwargs["auth_token"] = key
        return Anthropic(**kwargs)

    if config.api_contract == "openrouter":
        kwargs: dict[str, Any] = {"api_key": key}
        if timeout is not None:
            kwargs["timeout_ms"] = int(timeout * 1000)
        if max_retries is not None:
            from openrouter.utils.retries import RetryConfig
            kwargs["retry_config"] = RetryConfig(max_retries=max_retries)
        return OpenRouter(**kwargs)

    # openai
    base_url = base_url or config.base_url
    headers = dict(default_headers or {})
    query = dict(default_query or {})
    kwargs = {
        "base_url": base_url,
        "default_headers": headers or None,
        "default_query": query or None,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    if config.auth_mode == "x-api-key":
        kwargs["api_key"] = key
    else:
        kwargs["auth_token"] = key
    return OpenAI(**kwargs)


# client = make_client(Provider.ANTHROPIC)
model = os.environ.get("MODEL", "claude-haiku-4-5")
client = make_client(Provider.OPENROUTER)
current_provider = Provider.OPENROUTER 

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


def chat(messages, system=None, temperature=1.0, stop_sequences=[]):
    if isinstance(client, Anthropic):
        params = {
            "model": model,
            "max_tokens": 8000,
            "messages": messages,
            "temperature": temperature,
            "stop_sequences": stop_sequences,
            "thinking": {"type": "disabled"},
        }
        if system:
            params["system"] = system
        message = client.messages.create(**params)
        return "".join(block.text for block in message.content if block.type == "text")

    if isinstance(client, OpenRouter):
        from openrouter.components.chatrequest import Reasoning
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": 8000,
            "messages": messages,
            "temperature": temperature,
            "reasoning": Reasoning(effort="none"),
        }
        if system:
            params["messages"] = [{"role": "system", "content": system}] + messages
        if stop_sequences:
            params["stop"] = stop_sequences
        response = client.chat.send(**params)
        return _extract_text(response.choices[0].message.content)

    # OpenAI
    params = {
        "model": model,
        "max_tokens": 8000,
        "messages": messages,
        "temperature": temperature,
    }
    if system:
        params["system"] = system
    if stop_sequences:
        params["stop"] = stop_sequences
    completion = client.chat.completions.create(**params)
    return _extract_text(completion.choices[0].message.content)
