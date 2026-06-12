"""Thin put/get over a boto3 S3 client.

Deliberately un-abstracted: no pluggable backend, no local-dir mode. In
production :func:`_client` returns a real ``boto3`` client; tests monkeypatch
:func:`_client` with an in-memory fake and assert the put round-trips. That is
the whole transport layer — everything richer (multipart, retries, ACLs) is
out of scope for this command.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class S3Target:
    """An ``s3://bucket/prefix`` destination, split into bucket + key prefix."""

    bucket: str
    prefix: str

    def key_for(self, name: str) -> str:
        """Join ``name`` onto the prefix as an S3 key (no leading slash)."""
        if not self.prefix:
            return name
        return f"{self.prefix.rstrip('/')}/{name}"


def parse_s3_url(url: str) -> S3Target:
    """Parse ``s3://bucket/prefix/...`` into an :class:`S3Target`.

    Raises ``ValueError`` on a non-``s3`` scheme or a missing bucket so the
    caller can report a clean user error rather than upload into the void.
    """
    parsed = urlparse(url)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(
            f"Not an S3 URL: {url!r}. Expected the form s3://bucket/prefix"
        )
    return S3Target(bucket=parsed.netloc, prefix=parsed.path.lstrip("/"))


def _client():
    """Return a boto3 S3 client. Patched out in tests; real in production."""
    import boto3

    return boto3.client("s3")


def put_bytes(client, target: S3Target, name: str, data: bytes) -> str:
    """Upload ``data`` under ``target``'s prefix as object ``name``.

    Returns the full ``s3://`` URL the object landed at.
    """
    key = target.key_for(name)
    client.put_object(Bucket=target.bucket, Key=key, Body=data)
    return f"s3://{target.bucket}/{key}"


def put_file(client, target: S3Target, name: str, path) -> str:
    """Read ``path`` and upload it under ``target`` as object ``name``."""
    from pathlib import Path

    return put_bytes(client, target, name, Path(path).read_bytes())
