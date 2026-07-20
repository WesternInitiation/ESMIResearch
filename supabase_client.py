"""Supabase client factory for the ESMI Streamlit app."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import streamlit as st


class SupabaseNotConfiguredError(RuntimeError):
    """Raised when Streamlit secrets for Supabase are missing."""


class SupabaseImportError(RuntimeError):
    """Raised when the PyPI supabase package is unavailable."""


def _import_supabase():
    """Import the real PyPI package, with a clear error if it fails."""
    try:
        from supabase import create_client
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SupabaseImportError(
            "Could not import the PyPI 'supabase' package in this Python "
            "environment. Install it into the SAME interpreter that runs "
            "Streamlit, e.g.\n"
            "  C:\\ProgramData\\Anaconda3\\python.exe -m pip install -U supabase\n"
            "Also ensure there is no local folder named `supabase/` in the "
            "project (schema lives in `supabase_schema/`)."
        ) from exc
    return create_client


def supabase_package_available() -> bool:
    try:
        _import_supabase()
        return True
    except SupabaseImportError:
        return False


def supabase_configured() -> bool:
    """Return True when required Supabase secrets are present."""
    try:
        secrets = st.secrets["supabase"]
    except Exception:
        return False
    url = str(secrets.get("url", "")).strip()
    key = str(secrets.get("service_role_key", "")).strip()
    return bool(url and key)


def supabase_ready() -> bool:
    """True when both the package and secrets are available."""
    return supabase_configured() and supabase_package_available()


def get_supabase_settings() -> dict[str, str]:
    """Read and validate Supabase settings from st.secrets."""
    if not supabase_configured():
        raise SupabaseNotConfiguredError(
            "Supabase secrets are not configured. Add [supabase] url, "
            "service_role_key, and bucket to .streamlit/secrets.toml "
            "(or Streamlit Community Cloud secrets)."
        )
    secrets: dict[str, Any] = st.secrets["supabase"]
    url = str(secrets["url"]).strip().rstrip("/")
    key = str(secrets["service_role_key"]).strip()
    bucket = str(secrets.get("bucket", "esmi-images")).strip() or "esmi-images"
    return {"url": url, "service_role_key": key, "bucket": bucket}


@lru_cache(maxsize=1)
def _cached_client(url: str, key: str):
    create_client = _import_supabase()
    return create_client(url, key)


def get_supabase_client():
    """Return a shared Supabase client using the service role key."""
    settings = get_supabase_settings()
    return _cached_client(settings["url"], settings["service_role_key"])


def get_storage_bucket() -> str:
    return get_supabase_settings()["bucket"]
