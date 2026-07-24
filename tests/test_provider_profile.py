from __future__ import annotations

from onnx_graph_diff.providers import ProviderProfile, resolve_provider


def _profile() -> ProviderProfile:
    return ProviderProfile(
        name="gw",
        default_model="m",
        default_base_url="https://gw/v1",
        api_key_env="PRIMARY_KEY",
        api_key_env_fallbacks=("LEGACY_KEY",),
    )


def test_key_env_candidates_dedupe_and_order():
    profile = _profile()
    # Override first, then primary, then fallbacks, with duplicates removed.
    assert profile.key_env_candidates("OVERRIDE") == ("OVERRIDE", "PRIMARY_KEY", "LEGACY_KEY")
    assert profile.key_env_candidates("PRIMARY_KEY") == ("PRIMARY_KEY", "LEGACY_KEY")
    assert profile.key_env_candidates() == ("PRIMARY_KEY", "LEGACY_KEY")


def test_resolve_api_key_prefers_explicit_value(monkeypatch):
    monkeypatch.setenv("PRIMARY_KEY", "from-env")
    assert _profile().resolve_api_key(explicit="explicit-value") == "explicit-value"


def test_resolve_api_key_reads_primary_then_fallback(monkeypatch):
    monkeypatch.delenv("PRIMARY_KEY", raising=False)
    monkeypatch.setenv("LEGACY_KEY", "legacy-value")
    assert _profile().resolve_api_key() == "legacy-value"

    monkeypatch.setenv("PRIMARY_KEY", "primary-value")
    assert _profile().resolve_api_key() == "primary-value"


def test_resolve_api_key_honours_override_env(monkeypatch):
    monkeypatch.delenv("PRIMARY_KEY", raising=False)
    monkeypatch.setenv("CUSTOM_KEY", "custom-value")
    assert _profile().resolve_api_key(override_env="CUSTOM_KEY") == "custom-value"


def test_resolve_api_key_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("PRIMARY_KEY", raising=False)
    monkeypatch.delenv("LEGACY_KEY", raising=False)
    assert _profile().resolve_api_key() is None


def test_openai_compatible_keeps_openai_key_fallback():
    profile = resolve_provider("openai-compatible")
    assert profile.key_env_candidates() == ("LLM_API_KEY", "OPENAI_API_KEY")
