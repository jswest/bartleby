"""Bartleby, the Scrivener - A PDF processing tool."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bartleby")
except PackageNotFoundError:  # running from a bare source tree, not installed
    __version__ = "0.0.0+unknown"
