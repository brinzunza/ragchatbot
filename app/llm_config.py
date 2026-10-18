"""
LLM Configuration Module

This module provides centralized LLM initialization across three providers:
- OpenAI, if OPENAI_API_KEY is set in .env
- Anthropic (Claude), if ANTHROPIC_API_KEY is set in .env
- Otherwise (or by choice), local Ollama

Which provider answers is LLM_PROVIDER in .env (openai | anthropic | ollama).
When that's unset - or names a provider with no key - the first of OpenAI,
Anthropic that has a key wins, and with no key at all it's Ollama. Keys and
the choice are read at call time, so changing them applies without a restart.
"""

import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

# Load environment variables from .env file
load_dotenv()

_PROVIDER_ENV = "LLM_PROVIDER"

# Order matters: it's the precedence when no provider is chosen explicitly.
CLOUD_PROVIDERS = {
    "openai": {
        "label": "OpenAI",
        "key_env": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
        "default_model": "gpt-4o-mini",
    },
    "anthropic": {
        "label": "Claude",
        "key_env": "ANTHROPIC_API_KEY",
        "model_env": "ANTHROPIC_MODEL",
        "default_model": "claude-haiku-5-5",
    },
}
ALL_PROVIDERS = tuple(CLOUD_PROVIDERS) + ("ollama",)


def _env_file_path() -> str:
    """Path to the .env file, creating one alongside it if none exists yet."""
    path = find_dotenv(usecwd=True)
    return path or os.path.join(os.path.dirname(__file__), "..", ".env")


def _write_env_value(name: str, value: str) -> None:
    """Set NAME=value in .env (replacing any existing line) and in this process."""
    path = _env_file_path()
    lines = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    replaced = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{name}="):
            lines[i] = f"{name}={value}\n"
            replaced = True
            break
    if not replaced:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(f"{name}={value}\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    os.environ[name] = value


def _remove_env_value(name: str) -> None:
    """Remove NAME from .env and from this process."""
    path = _env_file_path()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        lines = [line for line in lines if not line.strip().startswith(f"{name}=")]
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    os.environ.pop(name, None)


def _check_provider(provider: str, allow_ollama: bool = False) -> None:
    if provider not in CLOUD_PROVIDERS and not (allow_ollama and provider == "ollama"):
        raise ValueError(f"unknown provider: {provider!r}")


def _key_for(provider: str) -> str | None:
    return os.getenv(CLOUD_PROVIDERS[provider]["key_env"]) or None


def get_active_provider() -> str:
    """The provider that will answer: 'openai', 'anthropic' or 'ollama'."""
    chosen = (os.getenv(_PROVIDER_ENV) or "").strip().lower()
    if chosen == "ollama":
        return "ollama"
    if chosen in CLOUD_PROVIDERS and _key_for(chosen):
        return chosen
    for provider in CLOUD_PROVIDERS:
        if _key_for(provider):
            return provider
    return "ollama"


def _mask(key: str) -> str:
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}{'*' * (len(key) - 8)}{key[-4:]}"


def get_masked_api_key(provider: str | None = None) -> str | None:
    """A provider's key masked for display (default: the active one), or None if unset."""
    provider = provider or get_active_provider()
    if provider not in CLOUD_PROVIDERS:
        return None
    key = _key_for(provider)
    return _mask(key) if key else None


def set_api_key(api_key: str, provider: str = "openai") -> None:
    """Persist a provider's key to .env and make that provider the active one."""
    _check_provider(provider)
    _write_env_value(CLOUD_PROVIDERS[provider]["key_env"], api_key)
    _write_env_value(_PROVIDER_ENV, provider)


def clear_api_key(provider: str | None = None) -> None:
    """
    Remove a provider's key from .env (default: the active provider's). If it
    was the chosen provider, the choice is dropped too, so the next provider
    with a key - or Ollama - takes over immediately.
    """
    provider = provider or get_active_provider()
    if provider == "ollama":
        return
    _check_provider(provider)
    _remove_env_value(CLOUD_PROVIDERS[provider]["key_env"])
    if (os.getenv(_PROVIDER_ENV) or "").strip().lower() == provider:
        _remove_env_value(_PROVIDER_ENV)


def set_active_provider(provider: str) -> None:
    """
    Choose which provider answers. Ollama is always available; a cloud
    provider needs its key saved first.

    Raises:
        ValueError: if the provider is unknown or has no key
    """
    _check_provider(provider, allow_ollama=True)
    if provider != "ollama" and not _key_for(provider):
        raise ValueError(f"add your {CLOUD_PROVIDERS[provider]['label']} API key first")
    _write_env_value(_PROVIDER_ENV, provider)


def get_providers_status() -> dict:
    """Per-provider key state and model, for the settings screen (Ollama needs no key)."""
    status = {}
    for provider, info in CLOUD_PROVIDERS.items():
        key = _key_for(provider)
        status[provider] = {
            "label": info["label"],
            "has_key": key is not None,
            "masked_key": _mask(key) if key else None,
            "model": os.getenv(info["model_env"], info["default_model"]),
        }
    status["ollama"] = {
        "label": "Ollama",
        "has_key": False,
        "masked_key": None,
        "model": os.getenv("OLLAMA_MODEL", "llama3.2:latest"),
    }
    return status


def get_llm(temperature: float = 0, streaming: bool = True, **kwargs) -> BaseChatModel:
    """
    Get an LLM instance for the active provider (see get_active_provider).

    Args:
        temperature: Model temperature (0-1, default 0)
        streaming: Whether to enable streaming responses (default True)
        **kwargs: Additional model-specific parameters. num_predict is the
            output-token cap (Ollama's name; Claude's max_tokens); num_ctx is
            Ollama-only and ignored elsewhere.

    Returns:
        Initialized LLM instance (ChatOpenAI, ChatAnthropic or ChatOllama)
    """
    provider = get_active_provider()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        info = CLOUD_PROVIDERS["openai"]
        model_name = os.getenv(info["model_env"], info["default_model"])

        print(f"🌐 Using OpenAI API with model: {model_name}")

        # Filter out Ollama-specific parameters that OpenAI doesn't support
        openai_kwargs = {k: v for k, v in kwargs.items()
                        if k not in ['num_predict', 'num_ctx']}

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            streaming=streaming,
            api_key=_key_for("openai"),
            **openai_kwargs
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        info = CLOUD_PROVIDERS["anthropic"]
        model_name = os.getenv(info["model_env"], info["default_model"])

        print(f"🌐 Using Anthropic API with model: {model_name}")

        anthropic_kwargs = {k: v for k, v in kwargs.items()
                            if k not in ['num_predict', 'num_ctx']}
        # Claude requires an explicit output cap
        anthropic_kwargs["max_tokens"] = kwargs.get("num_predict", 2000)

        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            streaming=streaming,
            api_key=_key_for("anthropic"),
            **anthropic_kwargs
        )

    # Local Ollama
    from langchain_ollama import ChatOllama

    model_name = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

    print(f"🏠 Using local Ollama with model: {model_name}")

    # Map common kwargs to Ollama-specific parameters
    ollama_kwargs = {}
    if 'num_predict' in kwargs:
        ollama_kwargs['num_predict'] = kwargs['num_predict']
    if 'num_ctx' in kwargs:
        ollama_kwargs['num_ctx'] = kwargs['num_ctx']

    return ChatOllama(
        model=model_name,
        temperature=temperature,
        streaming=streaming,
        **ollama_kwargs
    )


def get_model_info() -> dict:
    """
    Get information about the currently configured LLM.

    Returns:
        Dictionary with the active provider's name, model, location and id
    """
    provider = get_active_provider()
    if provider in CLOUD_PROVIDERS:
        info = CLOUD_PROVIDERS[provider]
        return {
            "provider": info["label"],
            "provider_id": provider,
            "model": os.getenv(info["model_env"], info["default_model"]),
            "location": "cloud",
        }
    return {
        "provider": "Ollama",
        "provider_id": "ollama",
        "model": os.getenv("OLLAMA_MODEL", "llama3.2:latest"),
        "location": "local",
    }
