/**
 * Generation-lifecycle contract tests.
 *
 * Contract: every started generation reaches a terminal state bearing its own
 * generationId; evidence from one generation can never satisfy a watcher of
 * another; and ONE shipped classifier decides terminal state for the app, both
 * harnesses, and these tests.
 *
 * The module tests below import and execute the shipped module directly.
 *
 * The "production wiring" section is weaker and says so: those are source
 * PRESENCE checks (imports exist, the choke point is assigned, calls appear).
 * They do not execute generate() and cannot catch a semantically broken call —
 * an independent review demonstrated a wrong-generation-id mutant passing them.
 * They guard against wholesale removal only; do not read them as proof of
 * lifecycle closure. Behavioral coverage of the wiring comes from the live
 * probes in tools/headless_smoke.mjs, which execute the real page.
 *
 * Run: node tests/test_generation_identity.mjs
 */
import {
  inProgressReceipt,
  failureReceipt,
  ensureTerminalReceipt,
  classifyGenerationState,
} from '../src/lib/generation-state.js';
import { readFileSync } from 'node:fs';

const results = [];
const check = (name, ok, detail = '') => results.push({ name, ok, detail });

// --- receipt constructors carry identity and terminality ---
{
  const r = inProgressReceipt(7);
  check('inProgressReceipt carries id + in-progress status',
        r.generationId === 7 && r.status === 'in-progress' && !!r.createdAt);
}
{
  const r = failureReceipt(7, 'embedding-unreachable', 'ECONNREFUSED');
  check('failureReceipt is terminal with phase and reason',
        r.generationId === 7 && r.status === 'failed'
        && r.phase === 'embedding-unreachable' && r.reason === 'ECONNREFUSED');
}

// --- ensureTerminalReceipt closes any generation left in-progress ---
{
  const closed = ensureTerminalReceipt(inProgressReceipt(3), 3);
  check('in-progress for the settled id becomes failed',
        closed.status === 'failed' && closed.generationId === 3);
}
{
  const real = { status: 'real', generationId: 3 };
  check('a terminal receipt is left unchanged',
        ensureTerminalReceipt(real, 3) === real);
}
{
  const other = inProgressReceipt(4);
  check('another generation\'s receipt is left unchanged',
        ensureTerminalReceipt(other, 3) === other);
}
{
  check('null receipt handled without throwing',
        ensureTerminalReceipt(null, 3) === null);
}

// --- the classifier: what a harness may accept ---
const S = (receipt, expectedId, canvasPresent = true) =>
  classifyGenerationState({ receipt, expectedId, canvasPresent });
{
  check('no receipt -> not done', !S(null, 1).done);
  check('stale generation -> not done',
        !S({ status: 'real', generationId: 1 }, 2).done);
  check('null expectedId -> not done (never universally fresh)',
        !S({ status: 'real', generationId: 1 }, null).done);
  check('in-progress -> not done',
        !S(inProgressReceipt(1), 1).done);
  const f = S(failureReceipt(1, 'embedding-http', '503'), 1);
  check('failed -> done, not ok, phase surfaced',
        f.done && !f.ok && f.phase === 'embedding-http');
  const inv = S({ status: 'invalid', generationId: 1, fallbackReason: 'joints: empty' }, 1);
  check('invalid -> done, not ok', inv.done && !inv.ok);
  check('real without canvas -> not done yet',
        !S({ status: 'real', generationId: 1 }, 1, false).done);
  const ok = S({ status: 'real', generationId: 1 }, 1, true);
  check('real with canvas -> done, ok', ok.done && ok.ok);
  const unk = S({ status: 'banana', generationId: 1 }, 1);
  check('unknown status -> done, not ok (fail loud, not hang)', unk.done && !unk.ok);
}

// --- production wiring uses the shipped module through one choke point ---
const mainSrc = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
const stripComments = (src) => src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '');
{
  check('main.js imports the shipped generation-state module',
        /from '\.\/lib\/generation-state\.js'/.test(mainSrc));
  check('main.js exposes the classifier choke point',
        /__kimodoGenerationState\s*=/.test(mainSrc));
  const genBody = stripComments(mainSrc.slice(mainSrc.indexOf('async function generate()')));
  const publishIdx = genBody.search(/__kimodoLastReceipt\s*=\s*inProgressReceipt\(/);
  const firstAwait = genBody.search(/\bawait\s/);
  check('generate() publishes an in-progress receipt before its first await',
        publishIdx !== -1 && publishIdx < firstAwait,
        publishIdx === -1 ? 'inProgressReceipt not used' : `publish at ${publishIdx}, first await at ${firstAwait}`);
  check('generate() has a terminal backstop (finally + ensureTerminalReceipt)',
        /finally\s*\{[\s\S]*?ensureTerminalReceipt/.test(genBody),
        'no finally-guard found');
}
for (const tool of ['headless_smoke.mjs', 'filmstrip_smoke.mjs']) {
  const src = readFileSync(new URL(`../tools/${tool}`, import.meta.url), 'utf8');
  check(`${tool} calls the shipped classifier choke point`,
        /__kimodoGenerationState\(/.test(src), 'no call found');
  check(`${tool} captures the prior generation id before clicking`,
        /prior(Generation)?Id/.test(src), 'no prior-id capture found');
}

const failed = results.filter(r => !r.ok);
for (const r of results) console.log(`${r.ok ? 'PASS' : 'FAIL'}  ${r.name}${!r.ok && r.detail ? `   [${r.detail}]` : ''}`);
console.log(`\n${results.length - failed.length}/${results.length} passed`);
process.exit(failed.length ? 1 : 0);
