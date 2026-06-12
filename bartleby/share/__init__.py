"""Corpus sharing: publish a findings-free copy of a corpus to S3.

``bartleby project publish <name> --to <s3-url>`` lives here. The package is
two thin pieces: :mod:`bartleby.share.s3` (a put/get wrapper over a real boto3
client) and :mod:`bartleby.share.publish` (the VACUUM-INTO copy, the bounded
session-layer strip on that copy, and the upload of the ``.db`` plus the
content-addressed original files). The source corpus is never mutated.
"""
