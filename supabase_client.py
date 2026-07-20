"""Supabase client factory for the ESMI Streamlit app."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import streamlit as st

try:
    from supabase import Client, create_client
except ImportError as exc:  # pragma: no cover - environment/package issues
    raise ImportError(
        "Could not import the PyPI 'supabase' package. Install it with "
        "`pip install supabase` and make sure this repo has no local folder "
        "named `supabase/` shadowing the package (schema lives in "
        "`supabase_schema/`)."
    ) from exc


class SupabaseNotConfiguredError(RuntimeError):
    """Raised when Streamlit secrets for Supabase are missing."""


def supabase_configured() -> bool:
    """Return True when required Supabase secrets are present."""
    try:
        secrets = st.secrets["supabase"]
    except Exception:
        return False
    url = str(secrets.get("url", "")).strip()
    key = str(secrets.get("service_role_key", "")).strip()
    return bool(url and key)


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
def _cached_client(url: str, key: str) -> Client:
    return create_client(url, key)


def get_supabase_client() -> Client:
    """Return a shared Supabase client using the service role key."""
    settings = get_supabase_settings()
    return _cached_client(settings["url"], settings["service_role_key"])


def get_storage_bucket() -> str:
    return get_supabase_settings()["bucket"]
