/**
 * Generation lifecycle OWNER contract (review finding F1/F2 of the
 * motion-export slice).
 *
 * The owner is executable, not presence-checked: these tests drive the
 * shipped module with injected evidence sinks and deferred generations
 * resolved in BOTH orders, proving that an overlapping or superseded
 * invocation can never begin, republish, resurrect, or settle evidence it
 * does not own — the exact schedules the fresh-context review demonstrated
 * against the previous presence-only checks.
 */

import {
  createGenerationLifecycle,
  classifyMotionExport,
} from '../src/lib/generation-state.js';

let failures = 0;
function check(name, ok, detail = '') {
  if (ok) console.log(`  ok  ${name}`);
  else { failures++; console.error(`FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}

function makeStore() {
  const store = { receipt: null, motion: null };
  return {
    store,
    sinks: {
      setReceipt: (r) => { store.receipt = r; },
      setMotion: (m) => { store.motion = m; },
      getReceipt: () => store.receipt,
    },
  };
}

const realReceipt = (id) => ({ status: 'real', generationId: id });
const motionFor = (id) => ({
  generationId: id,
  motion: [Array.from({ length: 369 }, () => 0)],
  numFrames: 1,
});

// --- Single-flight gate ----------------------------------------------------

{
  const { store, sinks } = makeStore();
  const lifecycle = createGenerationLifecycle(sinks);
  const run1 = lifecycle.begin();
  check('first begin() starts generation 1 and installs in-progress evidence',
    run1?.generationId === 1 && store.receipt?.status === 'in-progress'
      && store.receipt?.generationId === 1 && store.motion === null,
    JSON.stringify(store.receipt));

  const run2 = lifecycle.begin();
  check('second begin() while in flight is rejected, not queued',
    run2 === null && lifecycle.activeId === 1, JSON.stringify({ run2, active: lifecycle.activeId }));

  check('rejected begin() did not disturb the active generation evidence',
    store.receipt?.generationId === 1 && store.receipt?.status === 'in-progress');
}

// --- Both completion orders with an (illegally obtained) stale handle ------

// Order A: stale run finishes AFTER the newer generation started.
{
  const { store, sinks } = makeStore();
  const lifecycle = createGenerationLifecycle(sinks);
  const runN = lifecycle.begin();
  // N settles (as failure), releasing the flight slot; N+1 begins.
  check('publishFailure by the owner is accepted',
    runN.publishFailure('embedding-http', '503') === true
      && store.receipt?.status === 'failed');
  runN.settle();
  const runN1 = lifecycle.begin();
  check('after settle, next begin() starts generation 2',
    runN1?.generationId === 2 && store.receipt?.generationId === 2);

  // N's retained handle tries to resurrect its evidence while N+1 is live.
  const republished = runN.publishSuccess(realReceipt(1), motionFor(1));
  check('superseded handle cannot republish receipt or motion',
    republished === false && store.receipt?.generationId === 2 && store.motion === null,
    JSON.stringify({ republished, receipt: store.receipt, motion: store.motion }));

  const settled = runN.settle();
  check('superseded handle cannot settle the newer generation',
    settled === false && store.receipt?.generationId === 2
      && store.receipt?.status === 'in-progress');

  check('watcher of generation 1 never sees resurrected usable motion',
    classifyMotionExport({ motion: store.motion, receipt: store.receipt, expectedId: 1 }).usable === false);
}

// Order B: newer generation completes first; stale handle then tries to
// overwrite the newer TERMINAL evidence.
{
  const { store, sinks } = makeStore();
  const lifecycle = createGenerationLifecycle(sinks);
  const runN = lifecycle.begin();
  runN.publishFailure('exception', 'lost the race');
  runN.settle();
  const runN1 = lifecycle.begin();
  check('generation 2 publishes real evidence',
    runN1.publishSuccess(realReceipt(2), motionFor(2)) === true
      && store.receipt?.status === 'real' && store.motion?.generationId === 2);
  runN1.settle();

  const overwrote = runN.publishFailure('late-failure', 'stale write');
  check('stale handle cannot overwrite newer terminal evidence',
    overwrote === false && store.receipt?.status === 'real'
      && store.receipt?.generationId === 2 && store.motion?.generationId === 2,
    JSON.stringify(store.receipt));

  check('generation 2 evidence classifies usable for its watcher',
    classifyMotionExport({ motion: store.motion, receipt: store.receipt, expectedId: 2 }).usable === true);
}

// --- Settle semantics ------------------------------------------------------

{
  const { store, sinks } = makeStore();
  const lifecycle = createGenerationLifecycle(sinks);
  const run = lifecycle.begin();
  // Run suspends and settles without any terminal publication: the backstop
  // must convert in-progress to failed, and the slot must free.
  const ok = run.settle();
  check('settling an unterminated run applies the terminal backstop',
    ok === true && store.receipt?.status === 'failed'
      && store.receipt?.generationId === 1 && lifecycle.activeId === null,
    JSON.stringify(store.receipt));

  check('slot is reusable after backstop settle',
    lifecycle.begin()?.generationId === 2);
}

{
  const { store, sinks } = makeStore();
  const lifecycle = createGenerationLifecycle(sinks);
  const run = lifecycle.begin();
  run.publishSuccess(realReceipt(1), motionFor(1));
  const ok = run.settle();
  check('settle preserves owner-published terminal-real evidence',
    ok === true && store.receipt?.status === 'real' && store.motion?.generationId === 1);
  check('settle returns ownership truth the caller can gate UI on',
    run.settle() === false, 'second settle by the same handle must not report ownership');
}

// --- main.js wiring: generation flow must route through the owner ----------

import { readFileSync } from 'node:fs';
const mainSrc = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');

check('main.js constructs the lifecycle owner over the window evidence globals',
  mainSrc.includes('createGenerationLifecycle'));

check('main.js no longer assigns generation evidence directly',
  !/generationCounter/.test(mainSrc)
    && !/__kimodoLastReceipt\s*=\s*(inProgressReceipt|failureReceipt|ensureTerminalReceipt)/.test(mainSrc),
  'direct counter/receipt assignments must route through the lifecycle owner');

process.exit(failures ? 1 : 0);
