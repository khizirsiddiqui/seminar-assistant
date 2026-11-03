import json
import logging
from functools import lru_cache
from pathlib import Path

from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    InitSettingsSource,
    SettingsConfigDict,
)

logger = logging.getLogger("seminar.config")

ENV_FILE = ".env"
SETTINGS_PATH = Path("data/settings.json")
SETTING_FIELDS = ("llm_base_url", "llm_api_key", "llm_model", "arxiv_timeout_seconds")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8")

    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    arxiv_timeout_seconds: int = 20

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            init_settings,
            env_settings,
            DotEnvSettingsSource(settings_cls, env_file=ENV_FILE, env_file_encoding="utf-8"),
            InitSettingsSource(settings_cls, load_saved_settings()),
            file_secret_settings,
        )


def load_saved_settings() -> dict:
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        logger.warning("Ignoring unreadable settings file %s", SETTINGS_PATH, exc_info=True)
        return {}
    if not isinstance(data, dict):
        logger.warning("Ignoring unreadable settings file %s", SETTINGS_PATH, exc_info=True)
        return {}
    saved = {}
    for field in SETTING_FIELDS:
        if field not in data:
            continue
        value = data[field]
        if field == "arxiv_timeout_seconds":
            if isinstance(value, int) and not isinstance(value, bool):
                saved[field] = value
        elif isinstance(value, str):
            saved[field] = value
    return saved


def save_saved_settings(values: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {field: values[field] for field in SETTING_FIELDS if field in values}
    tmp = SETTINGS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.chmod(0o600)
    tmp.replace(SETTINGS_PATH)
    get_settings.cache_clear()


def update_saved_settings(updates: dict, remove: tuple[str, ...] = ()) -> dict:
    saved = load_saved_settings()
    for field in remove:
        saved.pop(field, None)
    for field in SETTING_FIELDS:
        if field not in updates:
            continue
        value = updates[field]
        if field == "arxiv_timeout_seconds":
            if isinstance(value, int) and not isinstance(value, bool):
                saved[field] = value
        elif isinstance(value, str):
            saved[field] = value
    save_saved_settings(saved)
    return saved


def setting_sources() -> dict[str, str]:
    env = EnvSettingsSource(Settings)()
    dot = DotEnvSettingsSource(Settings, env_file=ENV_FILE, env_file_encoding="utf-8")()
    saved = load_saved_settings()
    sources = {}
    for field in SETTING_FIELDS:
        if field in env:
            sources[field] = "environment"
        elif field in dot:
            sources[field] = ".env"
        elif field in saved:
            sources[field] = "settings"
        else:
            sources[field] = "default"
    return sources


@lru_cache
def get_settings() -> Settings:
    return Settings()
