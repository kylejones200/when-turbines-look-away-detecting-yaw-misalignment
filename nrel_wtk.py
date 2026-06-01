"""NREL Wind Toolkit CSV fetch via config.yaml and environment secrets."""

from __future__ import annotations

import logging
import os
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"
DEFAULT_URL = (
    "https://developer.nrel.gov/api/wind-toolkit/v2/wind/wtk-bchrrr-v1-0-0-download.csv"
)
DEFAULT_API_KEY_ENV = "NREL_API_KEY"
DEFAULT_EMAIL_ENV = "NREL_EMAIL"
DEFAULT_LAT = 41.5
DEFAULT_LON = -93.5
DEFAULT_YEARS: list[int] = [2017]
DEFAULT_ATTRIBUTES = "windspeed_100m,winddirection_100m,temperature_100m"
DEFAULT_INTERVAL_MINUTES = "60"
DEFAULT_TIMEOUT_SECONDS = 120
WTK_TIMESTAMP_COLUMNS = ["Year", "Month", "Day", "Hour", "Minute"]
CSV_DATA_HEADER_PREFIX = "Year,"


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load YAML config and apply secrets from ``.env`` when present.

    Args:
        config_path: Path to ``config.yaml``. Defaults to the file beside this module.

    Returns:
        Parsed config dict, or an empty dict if the file does not exist.
    """
    path = config_path or DEFAULT_CONFIG_PATH
    if path.exists():
        load_dotenv(path.parent / ".env")
    else:
        load_dotenv()
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _nrel_settings(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("nrel", {})


def get_nrel_api_key(config: dict[str, Any]) -> str:
    """Return the NREL API key from the environment.

    Args:
        config: Loaded config containing ``nrel.api_key_env``.

    Returns:
        API key string.

    Raises:
        RuntimeError: If the configured environment variable is unset.
    """
    nrel = _nrel_settings(config)
    env_name = nrel.get("api_key_env", DEFAULT_API_KEY_ENV)
    key = os.environ.get(env_name)
    if not key:
        raise RuntimeError(
            f"{env_name} is not set. Copy .env.example to .env and add your key from "
            "https://developer.nrel.gov/signup/"
        )
    return key


def get_nrel_email(config: dict[str, Any]) -> str:
    """Return the optional NREL contact email from the environment.

    Args:
        config: Loaded config containing ``nrel.email_env``.

    Returns:
        Email string, or empty if unset.
    """
    nrel = _nrel_settings(config)
    env_name = nrel.get("email_env", DEFAULT_EMAIL_ENV)
    return os.environ.get(env_name, "")


def csv_column_names(config: dict[str, Any]) -> list[str]:
    """Build CSV column names from timestamp fields and configured attributes.

    Args:
        config: Loaded config containing ``nrel.attributes``.

    Returns:
        Column names matching the NREL direct-download CSV layout.
    """
    nrel = _nrel_settings(config)
    attrs = nrel.get("attributes", DEFAULT_ATTRIBUTES)
    attribute_names = [name.strip() for name in attrs.split(",")]
    return [*WTK_TIMESTAMP_COLUMNS, *attribute_names]


def _api_bool(value: bool) -> str:
    return "true" if value else "false"


def build_nrel_params(config: dict[str, Any], year: int) -> dict[str, str]:
    """Build NREL Wind Toolkit query parameters for one year.

    Secrets (API key, email) are read from the environment, not from config.

    Args:
        config: Loaded config with ``nrel`` site and request settings.
        year: Calendar year to request.

    Returns:
        Query parameter mapping for ``requests.get``.
    """
    nrel = _nrel_settings(config)
    lat = nrel.get("lat", DEFAULT_LAT)
    lon = nrel.get("lon", DEFAULT_LON)
    return {
        "api_key": get_nrel_api_key(config),
        "wkt": f"POINT({lon} {lat})",
        "attributes": nrel.get("attributes", DEFAULT_ATTRIBUTES),
        "names": str(year),
        "utc": _api_bool(nrel.get("utc", True)),
        "leap_day": _api_bool(nrel.get("leap_day", False)),
        "interval": str(nrel.get("interval", DEFAULT_INTERVAL_MINUTES)),
        "email": get_nrel_email(config),
    }


def _csv_body_start_index(lines: list[str]) -> int:
    """Return the line index after the NREL metadata header row."""
    for index, line in enumerate(lines):
        if line.startswith(CSV_DATA_HEADER_PREFIX):
            return index + 1
    return 0


def _parse_year_csv(text: str, config: dict[str, Any]) -> pd.DataFrame:
    """Parse one year of NREL direct-download CSV text into a dataframe."""
    lines = text.strip().split("\n")
    data_text = "\n".join(lines[_csv_body_start_index(lines) :])
    frame = pd.read_csv(
        StringIO(data_text),
        header=None,
        names=csv_column_names(config),
    )
    frame["time"] = pd.to_datetime(frame[WTK_TIMESTAMP_COLUMNS])
    return frame


def _fetch_year(
    year: int,
    *,
    url: str,
    config: dict[str, Any],
    timeout: int,
) -> pd.DataFrame | None:
    """Fetch and parse wind data for a single year."""
    params = build_nrel_params(config, year)
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Year %s fetch failed: %s", year, exc)
        return None
    frame = _parse_year_csv(response.text, config)
    logger.info("Fetched %s records for year %s", f"{len(frame):,}", year)
    return frame


def fetch_nrel_wind_data(
    config: dict[str, Any] | None = None,
    config_path: Path | None = None,
) -> pd.DataFrame | None:
    """Fetch hourly WTK data for all years listed under ``nrel.years``.

    Args:
        config: Pre-loaded config. Loaded from ``config_path`` when omitted.
        config_path: Used only when ``config`` is None.

    Returns:
        Combined dataframe sorted by ``time``, or None if every year failed.
    """
    if config is None:
        config = load_config(config_path)
    nrel = _nrel_settings(config)
    years: list[int] = nrel.get("years", DEFAULT_YEARS)
    url = nrel.get("url", DEFAULT_URL)
    timeout = int(nrel.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))

    frames: list[pd.DataFrame] = []
    for year in years:
        logger.info("Fetching year %s", year)
        frame = _fetch_year(year, url=url, config=config, timeout=timeout)
        if frame is not None:
            frames.append(frame)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True).sort_values("time")
