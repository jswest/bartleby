# skill_runner closes the conn on every opened path, not just resolved sessions (issue #407)

> Source: [#407](https://github.com/jswest/bartleby/issues/407)

`run()` opens the DB (`open_db`), then resolves or creates the active session.
The teardown block gated *both* `conn.close()` and the audit `log_call` on a
single `if conn is not None and session_id is not None:`. That conflates two
unrelated concerns: closing the connection is a *resource* obligation, while the
audit write needs a resolved `session_id` as its FK target. If `open_db`
succeeded but session resolution then raised (a `BARTLEBY_SESSION_NAME` lookup
or `ensure_active_session` failure), `session_id` stayed `None`, the guard was
false, and the open `conn` leaked — every such failure dropping a connection.

Fix splits the guard into nested conditions: `if conn is not None:` always
closes, and `log_call` stays nested under `if session_id is not None:`. Closing
now happens on every path where the DB was opened, including the pre-session
failure; the audit row still requires a resolved session (it has nowhere to be
attributed otherwise) and that failure surfaces only in the stdout error
envelope. The JSON envelope and exit codes are unchanged.

The module docstring previously claimed "one `audit_logs` row per invocation
(success or failure)" unconditionally; that was never quite true (a failure
before session resolution writes none). It now states the row is written *once
a session is resolved*, and that the conn is closed on every opened path
regardless.

Smallest fix that fits: no helper, no `finally`, no behavior change beyond the
leak. Test `test_session_resolution_failure_closes_conn_keeps_envelope` wraps
`open_db` in a proxy that records `close()` (apsw's `Connection.close` is
read-only, so it can't be patched in place), forces session resolution to
raise, and asserts the conn was closed, the envelope is the standard
`INTERNAL_ERROR` shape, and no audit row was written. No schema change.
