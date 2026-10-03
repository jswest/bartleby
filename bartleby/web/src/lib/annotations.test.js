// Run with `npm test` (node:test — no web test harness). Pins locateAnchor to
// the Python contract in bartleby/db/annotations.py (#689) and checks the
// selection → anchor mapping the finding page relies on.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { locateAnchor, anchorFor, occurrenceIndex } from './annotations.js';

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

test('occurrenceIndex counts overlapping repeats like locateAnchor steps', () => {
  assert.equal(occurrenceIndex('aaaa', 'aa', 2), 2);
  assert.equal(occurrenceIndex(body, 'The cat', 39), 2);
});

test('anchorFor: minimal anchor that round-trips to the k-th occurrence', () => {
  assert.deepEqual(anchorFor(body, 'The cat', 0), { exact: 'The cat', prefix: null, suffix: null });
  for (const k of [1, 2]) {
    const a = anchorFor(body, 'The cat', k);
    const [start] = locateAnchor(body, a.exact, a.prefix, a.suffix);
    assert.equal(occurrenceIndex(body, 'The cat', start), k);
  }
  assert.equal(anchorFor(body, 'The cat', 3), null);
  assert.equal(anchorFor(body, 'not in body', 0), null);
});
