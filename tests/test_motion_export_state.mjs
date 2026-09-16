/**
 * Motion-export lifecycle contract.
 *
 * window.__kimodoLastMotion is generation evidence, exactly like the receipt:
 * it must never let a watcher read generation N's motion while generation N+1
 * is in flight, and it must never look usable when the receipt that vouches
 * for the same generation is missing, mismatched, or non-real.
 *
 * Module behavior is tested by importing and executing the shipped classifier.
 * main.js wiring (clear-at-start, publish-on-success, choke-point exposure)
 * is asserted by source presence, following test_generation_identity.mjs.
 */

import {
  classifyMotionExport,
} from '../src/lib/generation-state.js';
import { readFileSync } from 'node:fs';

let failures = 0;
function check(name, ok, detail = '') {
  if (ok) console.log(`  ok  ${name}`);
  else { failures++; console.error(`FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}

const realReceipt = (id) => ({ status: 'real', generationId: id });
// Rows are plain Array(369) — the representation the shipped producer builds
// and the receipt validator certifies (it requires Array.isArray per row).
const motion = (id, frames = 3) => ({
  generationId: id,
  motion: Array.from({ length: frames }, () => new Array(369).fill(0)),
  numFrames: frames,
});

// --- Behavioral contract -------------------------------------------------

{
  const r = classifyMotionExport({ motion: null, receipt: realReceipt(7), expectedId: 7 });
  check('null motion is unusable with reason no-motion',
    r.usable === false && r.reason === 'no-motion', JSON.stringify(r));
}

{
  const r = classifyMotionExport({ motion: motion(7), receipt: realReceipt(7), expectedId: null });
  check('missing expectedId never blesses motion (no universal freshness)',
    r.usable === false && r.reason === 'no-expected-id', JSON.stringify(r));
}

{
  const r = classifyMotionExport({ motion: motion(6), receipt: realReceipt(7), expectedId: 7 });
  check('motion from a previous generation is stale, not usable',
    r.usable === false && r.reason === 'stale-motion', JSON.stringify(r));
}

{
  const r = classifyMotionExport({ motion: motion(7), receipt: null, expectedId: 7 });
  check('motion without any receipt is unusable (no vouching authority)',
    r.usable === false && r.reason === 'receipt-motion-mismatch', JSON.stringify(r));
}

{
  const r = classifyMotionExport({ motion: motion(7), receipt: realReceipt(6), expectedId: 7 });
  check('motion whose receipt belongs to another generation is unusable',
    r.usable === false && r.reason === 'receipt-motion-mismatch', JSON.stringify(r));
}

{
  const r = classifyMotionExport({ motion: motion(7), receipt: { status: 'in-progress', generationId: 7 }, expectedId: 7 });
  check('in-progress receipt does not vouch for motion',
    r.usable === false && r.reason === 'receipt-not-real', JSON.stringify(r));
}

{
  const r = classifyMotionExport({ motion: motion(7), receipt: { status: 'failed', generationId: 7, phase: 'exception' }, expectedId: 7 });
  check('failed receipt does not vouch for motion',
    r.usable === false && r.reason === 'receipt-not-real', JSON.stringify(r));
}

{
  const bad = motion(7);
  bad.motion = [];
  const r = classifyMotionExport({ motion: bad, receipt: realReceipt(7), expectedId: 7 });
  check('empty motion rows are malformed, not usable',
    r.usable === false && r.reason === 'malformed-motion', JSON.stringify(r));
}

{
  const bad = motion(7, 3);
  bad.numFrames = 5;
  const r = classifyMotionExport({ motion: bad, receipt: realReceipt(7), expectedId: 7 });
  check('numFrames disagreeing with row count is malformed',
    r.usable === false && r.reason === 'malformed-motion', JSON.stringify(r));
}

{
  const r = classifyMotionExport({ motion: motion(7), receipt: realReceipt(7), expectedId: 7 });
  check('matching real receipt + matching motion is usable',
    r.usable === true, JSON.stringify(r));
}

// Mutation resistance: a classifier that ignores expectedId and just compares
// motion.generationId === receipt.generationId would pass most cases above.
// This case kills that mutant: both artifacts agree with each other but belong
// to a generation the watcher is NOT watching.
{
  const r = classifyMotionExport({ motion: motion(6), receipt: realReceipt(6), expectedId: 7 });
  check('mutant-killer: internally-consistent but wrong-generation pair is not usable',
    r.usable === false, JSON.stringify(r));
}

// --- main.js wiring (source presence) ------------------------------------
// Presence checks here are ROUTING assertions only; the lifecycle behavior
// itself (clear-at-start, ownership-gated publication, both completion
// orders) is executed against the shipped owner in
// tests/test_generation_lifecycle.mjs.

const mainSrc = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');

check('main.js wires the window evidence globals into the lifecycle owner',
  mainSrc.includes('createGenerationLifecycle')
    && /setMotion:\s*\(m\)\s*=>\s*\{\s*window\.__kimodoLastMotion\s*=\s*m/.test(mainSrc));

check('main.js publishes success evidence only through the owner handle',
  /run\.publishSuccess\(receipt,\s*motion\)/.test(mainSrc)
    && !/window\.__kimodoLastMotion\s*=\s*\{/.test(mainSrc));

check('main.js binds the producer generation to the lifecycle generation id',
  /producer\.generate\(\{[\s\S]*?generationId,/.test(mainSrc),
  'producer.generate must receive the owner-issued generationId');

check('main.js exposes the motion classifier choke point',
  /__kimodoMotionState\s*=/.test(mainSrc) && mainSrc.includes('classifyMotionExport'));

process.exit(failures ? 1 : 0);
