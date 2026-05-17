from __future__ import annotations
from dotenv import load_dotenv
from dataclasses import dataclass
from anthropic import Anthropic
from openai import OpenAI
import os
from enum import StrEnum
from typing import Any, Literal, Mapping, Union

load_dotenv()

class Provider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    OPENAI = "openai"

auth_mode = Literal["x-api-key", "bearer", "oauth"]
api_contract = Literal["anthropic", "openai"]

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
        api_contract="openai",
    ),
    Provider.OPENAI: ProviderConfig(
        base_url="https://api.openai.com/v1",
        auth_mode="bearer",
        env_key="OPENAI_API_KEY",
        api_contract="openai",
    ),
}

Client = Union[Anthropic, OpenAI]

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

    if config.api_contract == "anthropic":
        return Anthropic(**kwargs)
    else:
        return OpenAI(**kwargs)


# client = make_client(Provider.ANTHROPIC)
# model = os.environ.get("MODEL", "claude-haiku-4-5")
client = make_client(Provider.OPENROUTER)
current_provider = Provider.OPENROUTER
model = "deepseek/deepseek-v4-flash"

def add_user_message(messages, text):
    user_message = {"role": "user", "content": text}
    messages.append(user_message)


def add_assistant_message(messages, text):
    assistant_message = {"role": "assistant", "content": text}
    messages.append(assistant_message)


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
    else:
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

        # ugly hack but needed to disable thinking when using open router
        if  client.base_url == PROVIDERS[Provider.OPENROUTER].base_url:
            params["extra_body"] = {"include_reasoning": False}

        completion = client.chat.completions.create(**params)
        return completion.choices[0].message.content or ""
