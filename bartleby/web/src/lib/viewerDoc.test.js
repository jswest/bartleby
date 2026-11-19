// Run with `npm test` (node:test). Regression guard for #754: the passage
// matcher's fallback dropped leading lines until even a one-word remainder
// matched, so a citation could confidently highlight the wrong text.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { locatePassage } from './viewerDoc.js';

test('locatePassage: no highlight from a tiny remainder or a word fragment (#754)', () => {
  assert.equal(locatePassage('In 2019 the court held 5 things', 'H1\nH2\nH3\n5'), null);
  assert.equal(locatePassage('The other thing', 'he'), null);
});
