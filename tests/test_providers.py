from __future__ import annotations

import pytest

from onnx_graph_diff.llm_client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    LLMClient,
    LLMConfig,
    QwenClient,
    QwenConfig,
)
from onnx_graph_diff.providers import (
    DEFAULT_PROVIDER,
    ProviderProfile,
    profile_with_overrides,
    provider_from_env,
    register_provider,
    resolve_provider,
)


def test_default_provider_is_dashscope_backed():
    profile = resolve_provider(None)
    assert profile.name == "dashscope"
    assert profile.default_model == DEFAULT_MODEL
    assert profile.default_base_url == DEFAULT_BASE_URL
    assert profile.supports_thinking_stream is True


def test_qwen_is_an_alias_for_dashscope():
    assert resolve_provider("qwen") is resolve_provider("dashscope")


def test_openai_provider_profile():
    profile = resolve_provider("openai")
    assert profile.name == "openai"
    assert profile.default_base_url == "https://api.openai.com/v1"
    assert profile.api_key_env == "OPENAI_API_KEY"
    assert profile.supports_thinking_stream is False


def test_resolve_provider_is_case_insensitive():
    assert resolve_provider("OpenAI").name == "openai"


def test_resolve_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        resolve_provider("does-not-exist")


def test_backward_compatible_aliases():
    assert QwenClient is LLMClient
    assert QwenConfig is LLMConfig
    assert DEFAULT_PROVIDER == "dashscope"


def test_dashscope_client_instantiates_from_config(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    client = LLMClient(LLMConfig(api_key="fake-key"))
    assert client.profile.name == "dashscope"
    assert client.model == DEFAULT_MODEL
    assert client.base_url == DEFAULT_BASE_URL


def test_openai_client_instantiates_with_provider_defaults():
    client = LLMClient(LLMConfig(provider="openai", api_key="fake-key"))
    assert client.profile.name == "openai"
    assert client.model == "gpt-4o-mini"
    assert client.base_url == "https://api.openai.com/v1"


def test_openai_compatible_requires_a_model():
    with pytest.raises(RuntimeError, match="No model configured"):
        LLMClient(LLMConfig(provider="openai-compatible", api_key="fake-key"))


def test_openai_compatible_accepts_custom_endpoint():
    client = LLMClient(
        LLMConfig(
            provider="openai-compatible",
            model="llama-3.1-70b",
            base_url="http://localhost:11434/v1",
            api_key="fake-key",
        )
    )
    assert client.model == "llama-3.1-70b"
    assert client.base_url == "http://localhost:11434/v1"


def test_model_and_base_url_overrides_apply():
    client = LLMClient(LLMConfig(provider="openai", model="gpt-4.1", base_url="https://proxy/v1", api_key="fake-key"))
    assert client.model == "gpt-4.1"
    assert client.base_url == "https://proxy/v1"


def test_dashscope_reads_legacy_env_var(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "from-env")
    client = LLMClient(LLMConfig())
    # No exception means the key was resolved from the environment.
    assert client.profile.name == "dashscope"


def test_openai_reads_openai_api_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "from-openai-env")
    client = LLMClient(LLMConfig(provider="openai"))
    assert client.profile.name == "openai"


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key is required"):
        LLMClient(LLMConfig(provider="openai"))


def test_custom_api_key_env_override(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MY_GATEWAY_KEY", "gateway-secret")
    client = LLMClient(
        LLMConfig(
            provider="openai-compatible",
            model="m",
            base_url="http://gw/v1",
            api_key_env="MY_GATEWAY_KEY",
        )
    )
    assert client.profile.api_key_env == "MY_GATEWAY_KEY"


def test_explicit_api_key_beats_environment(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
    # Explicit key should be accepted without error even if env is also present.
    client = LLMClient(LLMConfig(api_key="explicit-key"))
    assert client.profile.name == "dashscope"


def test_provider_from_env_precedence(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert provider_from_env(None) == DEFAULT_PROVIDER
    assert provider_from_env("openai") == "openai"
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    # Environment wins over the caller default but a passed flag should be handled
    # by the caller resolving env first; provider_from_env prefers the env var.
    assert provider_from_env(None) == "openai"
    assert provider_from_env("dashscope") == "openai"


def test_profile_with_overrides_returns_same_when_no_changes():
    profile = resolve_provider("openai")
    assert profile_with_overrides(profile) is profile


def test_key_env_candidates_order():
    profile = resolve_provider("dashscope")
    assert profile.key_env_candidates("CUSTOM") == ("CUSTOM", "DASHSCOPE_API_KEY")
    assert profile.key_env_candidates() == ("DASHSCOPE_API_KEY",)


def test_register_custom_provider(monkeypatch):
    profile = ProviderProfile(
        name="my-cloud",
        default_model="my-model",
        default_base_url="https://my.cloud/v1",
        api_key_env="MY_CLOUD_KEY",
    )
    register_provider(profile)
    monkeypatch.setenv("MY_CLOUD_KEY", "secret")
    client = LLMClient(LLMConfig(provider="my-cloud"))
    assert client.model == "my-model"
    assert client.base_url == "https://my.cloud/v1"
