from __future__ import annotations

import pytest

from onnx_graph_diff import cli
from onnx_graph_diff.llm_client import LLMClient


def test_parser_exposes_provider_and_endpoint_flags():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "left.onnx",
            "right.onnx",
            "--method",
            "llm",
            "--provider",
            "openai",
            "--model",
            "gpt-4.1",
            "--base-url",
            "https://proxy/v1",
            "--api-key-env",
            "MY_KEY",
        ]
    )
    assert args.provider == "openai"
    assert args.model == "gpt-4.1"
    assert args.base_url == "https://proxy/v1"
    assert args.api_key_env == "MY_KEY"


def test_defaults_leave_provider_unset_for_env_resolution():
    parser = cli.build_parser()
    args = parser.parse_args(["left.onnx", "right.onnx"])
    assert args.provider is None
    assert args.model is None
    assert args.base_url is None


def test_llm_config_uses_selected_provider(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    parser = cli.build_parser()
    args = parser.parse_args(["l.onnx", "r.onnx", "--provider", "openai", "--seed", "7"])
    config = cli._llm_config(args)
    assert config.provider == "openai"
    assert config.seed == 7
    # The config resolves to the OpenAI profile without needing a live key.
    assert config.resolve_profile().name == "openai"


def test_llm_config_respects_llm_provider_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    parser = cli.build_parser()
    args = parser.parse_args(["l.onnx", "r.onnx"])
    config = cli._llm_config(args)
    assert config.provider == "openai"


def test_llm_config_flag_overrides_env(monkeypatch):
    # provider_from_env prefers the environment variable; the CLI passes the flag
    # as the default, so setting the env still wins unless unset.
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    parser = cli.build_parser()
    args = parser.parse_args(["l.onnx", "r.onnx", "--provider", "dashscope"])
    assert cli._llm_config(args).provider == "dashscope"


def test_cli_builds_openai_client_offline(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    parser = cli.build_parser()
    args = parser.parse_args(["l.onnx", "r.onnx", "--method", "llm", "--provider", "openai"])
    client = LLMClient(cli._llm_config(args))
    assert client.profile.name == "openai"
    assert client.model == "gpt-4o-mini"


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.build_parser().parse_args(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--provider" in out
    assert "--method" in out
