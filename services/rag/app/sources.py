"""File sources for ingestion.

Production pulls the uploaded PDF from Supabase private Storage using the
service-role key. The CLI and tests read local paths. Both return raw bytes so
the ingestion pipeline is source-agnostic.
"""
from __future__ import annotations

from pathlib import Path

import httpx

from .config import settings


def fetch_local(path: str) -> bytes:
    return Path(path).read_bytes()


def fetch_supabase_storage(storage_path: str) -> bytes:
    """Download an object from the private `kb` bucket with the service key."""
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise RuntimeError("Supabase storage not configured (url/service key).")
    url = (
        f"{settings.supabase_url}/storage/v1/object/"
        f"{settings.storage_bucket}/{storage_path}"
    )
    resp = httpx.get(
        url,
        headers={"Authorization": f"Bearer {settings.supabase_service_role_key}"},
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.content
