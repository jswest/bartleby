// Run with `npm test` (node:test — no web test harness). Pins locateAnchor to
// the Python contract in bartleby/db/annotations.py (#689) and checks the
// selection → anchor mapping the finding page relies on.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { locateAnchor, splitsMarker, bracketSpan, renderedSpan, anchorFromRendered } from './annotations.js';

const body = 'The cat sat. A dog ran. The cat slept. The cat sat again.';

test('locateAnchor: first occurrence, null on no match, empty exact throws', () => {
  assert.deepEqual(locateAnchor(body, 'The cat'), [0, 7]);
  assert.equal(locateAnchor(body, 'a bird'), null);
  assert.throws(() => locateAnchor(body, ''));
});

test('locateAnchor: prefix/suffix pick the bracketed occurrence', () => {
  assert.deepEqual(locateAnchor(body, 'The cat', 'ran. ', null), [24, 31]);
  assert.deepEqual(locateAnchor(body, 'The cat', null, ' sat again'), [39, 46]);
  assert.deepEqual(locateAnchor(body, 'The cat', 'slept. ', ' sat'), [39, 46]);
  assert.equal(locateAnchor(body, 'The cat', 'nope', null), null);
  // A prefix longer than the text before the match must not wrap around
  // (Python's startswith with a negative index would; the helper guards it).
  assert.equal(locateAnchor('ab', 'a', 'xb', null), null);
  // Empty prefix/suffix count as absent.
  assert.deepEqual(locateAnchor(body, 'The cat', '', ''), [0, 7]);
});

test('splitsMarker: a boundary strictly inside [^…] or ](…) splits; containing one is fine', () => {
  const b = 'Revenue rose [^chunk:1] in [Q3](https://x.com/q3) overall.';
  const at = (exact) => { const i = b.indexOf(exact); return [i, i + exact.length]; };
  assert.equal(splitsMarker(b, ...at('rose [^chunk:1')), true); // end inside [^…]
  assert.equal(splitsMarker(b, ...at('chunk:1] in')), true); // start inside [^…]
  assert.equal(splitsMarker(b, ...at('x.com')), true); // inside ](…)
  assert.equal(splitsMarker(b, ...at('Q3](https')), true); // end inside ](…)
  assert.equal(splitsMarker(b, ...at('rose [^chunk:1] in')), false); // contains the marker
  assert.equal(splitsMarker(b, ...at('[^chunk:1]')), false); // exactly the marker
  assert.equal(splitsMarker(b, ...at('Q3')), false); // link text, touching `](`
  assert.equal(splitsMarker(b, ...at('Revenue rose')), false);
});

// A toy stand-in for the page's render round-trip: link text survives, link
// targets vanish, a citation marker becomes a glyph + ordinal.
const toyRender = (md) => md.replace(/\[([^\]^]*)\]\([^)]*\)/g, '$1').replace(/\[\^[^\]]*\]/g, '¶1');
const placer = (raw) => (s, e) => renderedSpan(toyRender(bracketSpan(raw, s, e)), toyRender(raw));

test('renderedSpan traces a raw span through rendering, null when it does not survive', () => {
  const raw = 'See [the report](https://x.com/report) now.';
  const i = raw.indexOf('the report');
  assert.deepEqual(placer(raw)(i, i + 10), [4, 14]);
  const u = raw.indexOf('x.com'); // inside the dropped link target
  assert.equal(placer(raw)(u, u + 5), null);
});

test('anchorFromRendered: anchors the selected repeat from rendered context, not the URL', () => {
  const raw = 'See [the report](https://x.com/report) and the report says the report is late.';
  const rendered = toyRender(raw);
  const second = rendered.indexOf('report', rendered.indexOf('report') + 1);
  const a = anchorFromRendered(raw, rendered, second, 'report', placer(raw));
  assert.ok(a && !a.prefix.includes(']('), JSON.stringify(a));
  const [s] = locateAnchor(raw, a.exact, a.prefix, a.suffix);
  assert.equal(s, raw.indexOf('report says'));
  for (const sel of [rendered.indexOf('report'), rendered.lastIndexOf('report')]) {
    const b = anchorFromRendered(raw, rendered, sel, 'report', placer(raw));
    assert.deepEqual(placer(raw)(...locateAnchor(raw, b.exact, b.prefix, b.suffix)), [sel, sel + 6]);
  }
});

test('anchorFromRendered: null when the selection is not verbatim in the raw body', () => {
  const raw = 'Revenue rose [^chunk:1] sharply.';
  const rendered = toyRender(raw);
  const quote = 'rose ¶1 sharply';
  assert.equal(anchorFromRendered(raw, rendered, rendered.indexOf(quote), quote, placer(raw)), null);
});
