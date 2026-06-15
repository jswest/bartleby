"""The artifact-fetch seam shared by every ``project import`` transport.

``import`` adopts the same artifact ``publish`` produces: a ``bartleby.db`` at
the prefix root plus the content-addressed originals under
``files/<file_hash><ext>``. The only thing that varies between transports is
*where those named blobs come from* — S3, or a local directory / ``file://``
URL. Everything downstream (the schema + embedding-model gates, the ``.db``
adoption, the ``file_path`` rewrite by ``file_hash``, ``--without-tags``) is
identical, so it runs against this one narrow seam.

An :class:`ArtifactSource` answers a single question: "give me the bytes of the
artifact named ``X``". A genuinely-absent artifact raises :class:`MissingArtifact`
(both transports normalize their own not-found signal to it) so the import layer
can treat "the artifact advertised a file it doesn't actually hold" uniformly as
a corrupt-artifact refusal. Any other failure (a transient S3 error, a disk
read error) propagates unchanged so the import aborts and tears down.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlparse

from botocore.exceptions import ClientError

from bartleby.share import s3


class MissingArtifact(Exception):
    """The source does not hold an artifact it was asked for (absent object)."""


class ArtifactSource(Protocol):
    """Fetch a published artifact's bytes by its name within the artifact set.

    ``name`` is the same relative name on every transport: ``bartleby.db`` for
    the database, ``files/<file_hash><ext>`` for an original.
    """

    def get_bytes(self, name: str) -> bytes:
        """Return the bytes of artifact ``name``; raise :class:`MissingArtifact`
        if it is genuinely absent."""
        ...


class S3Source:
    """An :class:`ArtifactSource` backed by an ``s3://bucket/prefix`` artifact."""

    def __init__(self, target: s3.S3Target, client):
        self._target = target
        self._client = client

    def get_bytes(self, name: str) -> bytes:
        try:
            return s3.get_bytes(self._client, self._target, name)
        except ClientError as e:
            if _is_missing_key(e):
                raise MissingArtifact(name) from e
            raise


class LocalSource:
    """An :class:`ArtifactSource` backed by a local directory of artifacts.

    Points straight at a directory holding what ``publish`` produces
    (``bartleby.db`` + ``files/<file_hash><ext>``). No download — the bytes are
    read off disk — but the same name-keyed contract as :class:`S3Source`, so
    the import path is identical.
    """

    def __init__(self, root: Path):
        self._root = root

    def get_bytes(self, name: str) -> bytes:
        path = self._root / name
        try:
            return path.read_bytes()
        except FileNotFoundError as e:
            raise MissingArtifact(name) from e


def _is_missing_key(err: ClientError) -> bool:
    """True if a boto3 ``ClientError`` is an absent-object (NoSuchKey / 404)."""
    code = str(err.response.get("Error", {}).get("Code", ""))
    return code in ("NoSuchKey", "404", "NoSuchBucket")


def local_root_from_url(from_url: str) -> Path:
    """Resolve a ``--from`` value to a local artifact directory, or raise.

    Accepts a plain local path (``/path/to/artifact-dir``) or a ``file://`` URL
    (``file:///path/to/artifact-dir``). Raises ``ValueError`` if the value is an
    ``s3://`` URL (the caller routes those to S3), or if the resolved directory
    does not exist / is not a directory.
    """
    parsed = urlparse(from_url)
    if parsed.scheme == "file":
        # file:///abs/path -> netloc empty, path is the absolute path; a
        # file://host/path form is not a local path we can read.
        if parsed.netloc not in ("", "localhost"):
            raise ValueError(
                f"Unsupported file:// host in {from_url!r}; expected "
                "file:///absolute/path (an empty or localhost host)."
            )
        root = Path(unquote(parsed.path))
    elif parsed.scheme == "":
        root = Path(from_url)
    else:
        raise ValueError(
            f"Not a local path or file:// URL: {from_url!r}."
        )
    root = root.expanduser()
    if not root.is_dir():
        raise ValueError(
            f"Local import source is not a directory: {root}. Expected a "
            "directory holding bartleby.db and files/<file_hash><ext> (what "
            "`project publish` produces)."
        )
    return root


def is_s3_url(from_url: str) -> bool:
    """True if ``from_url`` should be routed to the S3 transport."""
    return urlparse(from_url).scheme == "s3"
