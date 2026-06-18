# GH-0648: Serve evidence files buffered, not streamed

## Context

`bartleby serve` (the Vite dev server) crashed the entire dev process with an
uncaught `TypeError [ERR_INVALID_STATE]: Invalid state: ReadableStream is already
closed` whenever a file request was aborted mid-flight — iframe reloads, client
disconnects, early-closing probes. Root cause: the file route returned a streamed
`ReadableStream` body, and when the client aborted, SvelteKit-dev + undici tried
to `.close()` an already-closed stream, throwing unhandled → process exit.

## Decision

Replace `fs.statSync` + `fs.createReadStream` with a single `fs.readFileSync` into
a `Buffer`, and drop the manual `Content-Length` header (SvelteKit drains the
buffered body as a chunked response, so no Content-Length is sent — correct for
browsers and the iframe). All other logic is unchanged: MIME
lookup, `Content-Disposition: inline`, `X-Content-Type-Options: nosniff`, and the
`Content-Security-Policy: sandbox` branch for HTML files.

## Trade-offs

**Whole-file in memory.** Buffering means the full file is held in the Node process
for the duration of the request. This is acceptable because:

- The corpus is small local evidence documents (md/pdf/images) served by a local
  dev tool — not a high-concurrency production server.
- The streaming lifecycle was the source of the crash; removing it eliminates the
  failure mode entirely.
- Simplicity wins over marginal memory savings at this scale.
