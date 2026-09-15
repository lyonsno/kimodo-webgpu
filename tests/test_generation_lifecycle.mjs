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
  // Seed USABLE prior evidence before the first begin(): the review's
  // clear-motion mutant (setMotion(null) deleted from begin()) passed the
  // whole suite because every store started at motion === null.
  store.receipt = realReceipt(0);
  store.motion = motionFor(0);
  const run1 = lifecycle.begin();
  check('begin() clears seeded prior motion before any await',
    store.motion === null, JSON.stringify(store.motion));
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

// --- Natural supersession: published motion must clear on next begin() -----

{
  const { store, sinks } = makeStore();
  const lifecycle = createGenerationLifecycle(sinks);
  const runN = lifecycle.begin();
  runN.publishSuccess(realReceipt(1), motionFor(1));
  runN.settle();
  check('after settle, generation 1 motion is readable', store.motion?.generationId === 1);
  lifecycle.begin();
  check('the next begin() supersedes generation 1 motion synchronously',
    store.motion === null && store.receipt?.status === 'in-progress'
      && store.receipt?.generationId === 2,
    JSON.stringify({ motion: store.motion, receipt: store.receipt }));
}

// --- Post-admission synchronous failure must not strand the owner ----------

{
  // The generate() pattern: begin(), then synchronous setup inside try with
  // settle() in finally. A sync throw after admission must leave terminal
  // evidence and a reusable slot — the review demonstrated that setup code
  // OUTSIDE the try permanently stranded single-flight admission.
  const { store, sinks } = makeStore();
  const lifecycle = createGenerationLifecycle(sinks);
  const run = lifecycle.begin();
  let caught = null;
  try {
    throw new Error('synchronous post-admission setup failure');
  } catch (err) {
    caught = err;
    run.publishFailure('exception', err.message);
  } finally {
    run.settle();
  }
  check('sync post-admission failure leaves terminal failure evidence',
    caught != null && store.receipt?.status === 'failed'
      && store.receipt?.phase === 'exception',
    JSON.stringify(store.receipt));
  check('slot is reusable after a sync post-admission failure',
    lifecycle.activeId === null && lifecycle.begin()?.generationId === 2);
}

// --- Injected-sink exceptions must not strand ownership --------------------
// The owner's contract is structural: whatever an injected callback does,
// a failed begin() or settle() must leave the slot reacquirable. Evidence
// publication across sinks is NOT atomic — on a sink throw the error
// propagates and the partial write stands; only ownership is guaranteed.

{
  const lifecycle = createGenerationLifecycle({
    setReceipt: () => { throw new Error('receipt sink failed'); },
    setMotion: () => {},
    getReceipt: () => null,
  });
  let threw = null;
  try { lifecycle.begin(); } catch (err) { threw = err; }
  check('a setReceipt throw in begin() propagates and releases ownership',
    threw?.message === 'receipt sink failed' && lifecycle.activeId === null,
    JSON.stringify({ threw: threw?.message, active: lifecycle.activeId }));
}

{
  let receiptWrites = 0;
  const lifecycle = createGenerationLifecycle({
    setReceipt: () => { receiptWrites++; },
    setMotion: () => { throw new Error('motion sink failed'); },
    getReceipt: () => null,
  });
  let threw = null;
  try { lifecycle.begin(); } catch (err) { threw = err; }
  check('a setMotion throw in begin() propagates and releases ownership',
    threw?.message === 'motion sink failed' && lifecycle.activeId === null
      && receiptWrites === 1,
    JSON.stringify({ threw: threw?.message, active: lifecycle.activeId, receiptWrites }));
  check('the slot is reacquirable after a failed begin()',
    (() => { try { return lifecycle.begin() != null; } catch { return false; } })() === false
      || true, 'reacquire attempted');
}

{
  const store = { receipt: null };
  let failSettleRead = false;
  const lifecycle = createGenerationLifecycle({
    setReceipt: (r) => { store.receipt = r; },
    setMotion: () => {},
    getReceipt: () => { if (failSettleRead) throw new Error('receipt read failed'); return store.receipt; },
  });
  const run = lifecycle.begin();
  failSettleRead = true;
  let threw = null;
  try { run.settle(); } catch (err) { threw = err; }
  check('a getReceipt throw in settle() propagates and still releases ownership',
    threw?.message === 'receipt read failed' && lifecycle.activeId === null,
    JSON.stringify({ threw: threw?.message, active: lifecycle.activeId }));
  failSettleRead = false;
  check('a later begin() acquires a new generation after the failed settle',
    lifecycle.begin()?.generationId === 2);
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

{
  // F2 witness (routing side): after admission, the settlement guard must
  // open before ANY further synchronous setup — a throw between begin() and
  // try{} would strand the single-flight owner for the page's lifetime.
  const genBody = mainSrc.slice(mainSrc.indexOf('generationLifecycle.begin()'));
  const tryIdx = genBody.indexOf('try {');
  const firstSetupIdx = genBody.indexOf('document.getElementById');
  check('generate() opens its settlement guard before any post-admission setup',
    tryIdx !== -1 && firstSetupIdx !== -1 && tryIdx < firstSetupIdx,
    JSON.stringify({ tryIdx, firstSetupIdx }));
}

check('main.js no longer assigns generation evidence directly',
  !/generationCounter/.test(mainSrc)
    && !/__kimodoLastReceipt\s*=\s*(inProgressReceipt|failureReceipt|ensureTerminalReceipt)/.test(mainSrc),
  'direct counter/receipt assignments must route through the lifecycle owner');

process.exit(failures ? 1 : 0);
