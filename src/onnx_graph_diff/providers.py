"""Provider abstraction for the OpenAI-compatible chat backends used by the matchers.

The matchers only depend on the small ``LLMClient`` surface (``propose_match``,
``propose_agentic_match``, ``explain``). This module isolates the *provider*
specifics -- default model, base URL, which environment variable holds the API
key, and a couple of transport quirks -- behind a :class:`ProviderProfile`. New
backends are added by registering a profile rather than by editing the client.

Built-in providers:

``dashscope`` / ``qwen``
    Alibaba Cloud DashScope OpenAI-compatible endpoint (the original default).
    Key from ``DASHSCOPE_API_KEY``. Supports the DashScope ``enable_thinking``
    streaming extension.

``openai``
    Public OpenAI API (``https://api.openai.com/v1``). Key from ``OPENAI_API_KEY``.

``openai-compatible``
    Generic escape hatch for any OpenAI-compatible server (vLLM, Together,
    Groq, Ollama, a self-hosted gateway, ...). Base URL, model, and the API-key
    environment variable are all supplied by the caller.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class ProviderProfile:
    """Static description of an OpenAI-compatible chat provider."""

    name: str
    default_model: str
    default_base_url: str
    api_key_env: str
    #: Provider understands the DashScope ``enable_thinking`` extra-body flag and
    #: the associated streamed "thinking" responses. Only DashScope/Qwen does.
    supports_thinking_stream: bool = False
    #: Env vars accepted for the API key, in priority order. The first non-empty
    #: value wins. Always includes ``api_key_env``; ``dashscope`` also keeps the
    #: legacy fallback so existing setups keep working.
    api_key_env_fallbacks: tuple[str, ...] = field(default_factory=tuple)

    def key_env_candidates(self, override_env: str | None = None) -> tuple[str, ...]:
        """Ordered environment variables to read the API key from.

        ``override_env`` (e.g. from a ``--api-key-env`` flag) takes priority,
        then the provider's primary variable, then any fallbacks.
        """

        candidates: list[str] = []
        for env in (override_env, self.api_key_env, *self.api_key_env_fallbacks):
            if env and env not in candidates:
                candidates.append(env)
        return tuple(candidates)

    def resolve_api_key(self, explicit: str | None = None, override_env: str | None = None) -> str | None:
        """Resolve the API key from an explicit value or the environment."""

        if explicit:
            return explicit
        for env in self.key_env_candidates(override_env):
            value = os.getenv(env)
            if value:
                return value
        return None


# DashScope keeps the historical DASHSCOPE_API_KEY fallback so pre-existing
# configurations continue to work unchanged.
_DASHSCOPE = ProviderProfile(
    name="dashscope",
    default_model="qwen3.5-122b-a10b",
    default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key_env="DASHSCOPE_API_KEY",
    supports_thinking_stream=True,
)

_OPENAI = ProviderProfile(
    name="openai",
    default_model="gpt-4o-mini",
    default_base_url="https://api.openai.com/v1",
    api_key_env="OPENAI_API_KEY",
)

# Generic OpenAI-compatible endpoint. Model and base URL are placeholders that
# the caller is expected to override; the key env var defaults to a neutral
# name and can be pointed elsewhere with ``api_key_env`` on the config.
_OPENAI_COMPATIBLE = ProviderProfile(
    name="openai-compatible",
    default_model="",
    default_base_url="http://localhost:8000/v1",
    api_key_env="LLM_API_KEY",
    api_key_env_fallbacks=("OPENAI_API_KEY",),
)


_PROVIDERS: dict[str, ProviderProfile] = {
    "dashscope": _DASHSCOPE,
    "qwen": _DASHSCOPE,  # convenience alias for the DashScope-hosted Qwen models
    "openai": _OPENAI,
    "openai-compatible": _OPENAI_COMPATIBLE,
}

#: Provider used when nothing is specified. Preserves the original behavior.
DEFAULT_PROVIDER = "dashscope"

#: Provider names exposed to CLI/argparse choices.
PROVIDER_CHOICES: tuple[str, ...] = ("dashscope", "qwen", "openai", "openai-compatible")


def available_providers() -> tuple[str, ...]:
    """Return the registered provider names (including aliases)."""

    return tuple(_PROVIDERS)


def resolve_provider(name: str | None) -> ProviderProfile:
    """Look up a provider profile by name.

    The name is matched case-insensitively; ``None`` selects the default
    provider. ``LLM_PROVIDER`` is *not* read here -- callers decide precedence
    between flag and environment.
    """

    key = (name or DEFAULT_PROVIDER).strip().lower()
    try:
        return _PROVIDERS[key]
    except KeyError:
        known = ", ".join(sorted(set(_PROVIDERS)))
        raise ValueError(f"Unknown LLM provider {name!r}. Known providers: {known}") from None


def register_provider(profile: ProviderProfile) -> None:
    """Register (or override) a provider profile by its ``name``.

    Intended for downstream extension: define a :class:`ProviderProfile` for a
    bespoke gateway and register it before constructing the client.
    """

    _PROVIDERS[profile.name.strip().lower()] = profile


def provider_from_env(default: str | None = None) -> str:
    """Return the provider name from ``LLM_PROVIDER`` or ``default``.

    This is the single place that reads the environment variable so precedence
    stays consistent across the CLI and library entry points.
    """

    return os.getenv("LLM_PROVIDER") or default or DEFAULT_PROVIDER


def profile_with_overrides(
    profile: ProviderProfile,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
) -> ProviderProfile:
    """Return a copy of ``profile`` with the given fields overridden when set."""

    if not (model or base_url or api_key_env):
        return profile
    return replace(
        profile,
        default_model=model or profile.default_model,
        default_base_url=base_url or profile.default_base_url,
        api_key_env=api_key_env or profile.api_key_env,
    )
