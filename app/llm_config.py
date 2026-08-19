"""
LLM Configuration Module

This module provides centralized LLM initialization with automatic fallback:
- If OPENAI_API_KEY is set in .env, uses OpenAI API
- Otherwise, falls back to local Ollama

This allows the system to work seamlessly with both cloud and local LLMs.
"""

import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

# Load environment variables from .env file
load_dotenv()

_ENV_KEY = "OPENAI_API_KEY"


def _env_file_path() -> str:
    """Path to the .env file, creating one alongside it if none exists yet."""
    path = find_dotenv(usecwd=True)
    return path or os.path.join(os.path.dirname(__file__), "..", ".env")


def get_masked_api_key() -> str | None:
    """Return the current OpenAI key masked for display, or None if unset."""
    key = os.getenv(_ENV_KEY)
    if not key:
        return None
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}{'*' * (len(key) - 8)}{key[-4:]}"


def set_api_key(api_key: str) -> None:
    """Persist a new OpenAI key to .env and apply it immediately."""
    path = _env_file_path()
    lines = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    replaced = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{_ENV_KEY}="):
            lines[i] = f"{_ENV_KEY}={api_key}\n"
            replaced = True
            break
    if not replaced:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(f"{_ENV_KEY}={api_key}\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    os.environ[_ENV_KEY] = api_key


def clear_api_key() -> None:
    """Remove the OpenAI key from .env and fall back to Ollama immediately."""
    path = _env_file_path()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        lines = [line for line in lines if not line.strip().startswith(f"{_ENV_KEY}=")]
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    os.environ.pop(_ENV_KEY, None)


def get_llm(temperature: float = 0, streaming: bool = True, **kwargs) -> BaseChatModel:
    """
    Get an LLM instance with automatic OpenAI/Ollama fallback.

    If OPENAI_API_KEY is found in environment variables, returns OpenAI ChatGPT.
    Otherwise, returns local Ollama instance.

    Args:
        temperature: Model temperature (0-1, default 0)
        streaming: Whether to enable streaming responses (default True)
        **kwargs: Additional model-specific parameters

    Returns:
        Initialized LLM instance (either ChatOpenAI or ChatOllama)
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        # Use OpenAI API
        from langchain_openai import ChatOpenAI

        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        print(f"🌐 Using OpenAI API with model: {model_name}")

        # Filter out Ollama-specific parameters that OpenAI doesn't support
        openai_kwargs = {k: v for k, v in kwargs.items()
                        if k not in ['num_predict', 'num_ctx']}

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            streaming=streaming,
            api_key=api_key,
            **openai_kwargs
        )
    else:
        # Fallback to local Ollama
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
        Dictionary with model provider and model name
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        return {
            "provider": "OpenAI",
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "location": "cloud"
        }
    else:
        return {
            "provider": "Ollama",
            "model": os.getenv("OLLAMA_MODEL", "llama3.2:latest"),
            "location": "local"
        }
