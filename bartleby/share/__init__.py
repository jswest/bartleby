"""Corpus sharing: publish a findings-free copy of a corpus, and import it back.

``bartleby project publish <name> --to <s3-url>`` and
``bartleby project import <name> --from <source>`` live here. The pieces:
:mod:`bartleby.share.s3` (a put/get wrapper over a real boto3 client),
:mod:`bartleby.share.publish` (the VACUUM-INTO copy, the bounded session-layer
strip on that copy, and the upload of the ``.db`` plus the content-addressed
original files), :mod:`bartleby.share.source` (the artifact-fetch seam — S3 or a
local directory / ``file://`` URL — that import runs against), and
:mod:`bartleby.share.import_` (verify + adopt + land, transport-agnostic). The
source corpus is never mutated.
"""
