/**
 * Generation-freshness tests for the evidence harnesses.
 *
 * An independent review found that window.__kimodoLastReceipt is never cleared
 * when a new generation starts, so after one success the page holds both a
 * canvas and a `real` receipt. A second run's harness can terminate immediately
 * on the FIRST run's evidence, before the second generation produces anything.
 *
 * These tests model that page state directly — no browser, no weights, no GPU —
 * by exercising the terminal-state predicate the harnesses use.
 *
 * Run: node tests/test_generation_identity.mjs
 */
const results = [];
const check = (name, ok, detail = '') => results.push({ name, ok, detail });

// Minimal stand-in for the browser globals the harness predicate reads.
function makePage() {
  return {
    canvas: null,
    receipt: null,
    status: '',
    // What generate() must do at the START of a run for freshness to hold.
    beginGeneration(id) {
      this.currentGenerationId = id;
      this.receipt = null;          // required: supersede prior evidence
      this.status = 'Generating...';
    },
    finishGeneration(id, status = 'real') {
      this.canvas = { id };
      this.receipt = { status, generationId: id };
      this.status = `Generated (${id})`;
    },
  };
}

// The predicate under test: what a harness uses to decide a run is done.
// It must accept only evidence belonging to the generation it is watching.
function terminalState(page, expectedId) {
  const s = page.status || '';
  if (/error|failed|unusable|cannot reach|invalid/i.test(s)) return { done: true, ok: false };
  const r = page.receipt;
  if (page.canvas && r) {
    const belongs = expectedId === undefined || r.generationId === expectedId;
    if (!belongs) return { done: false };          // stale evidence: keep waiting
    return { done: true, ok: r.status === 'real' };
  }
  return { done: false };
}

// --- generation A succeeds ---
const page = makePage();
page.beginGeneration(1);
page.finishGeneration(1);
{
  const v = terminalState(page, 1);
  check('generation A completes on its own evidence', v.done && v.ok);
}

// --- generation B starts: stale canvas + stale receipt still present ---
{
  const expectedB = 2;
  page.beginGeneration(expectedB);          // B in flight, no result yet
  const v = terminalState(page, expectedB);
  check('generation B does NOT complete on A\'s evidence', !v.done,
        `done=${v.done} ok=${v.ok}`);
}

// --- generation B then fails at /embed ---
{
  page.status = 'Text embedding request failed: 500 Internal Server Error.';
  const v = terminalState(page, 2);
  check('generation B failure is detected as failure', v.done && !v.ok);
}

// --- generation B succeeds on its own evidence ---
{
  page.beginGeneration(2);
  page.finishGeneration(2);
  const v = terminalState(page, 2);
  check('generation B completes on its own evidence', v.done && v.ok);
}

// --- a receipt from an older generation must never satisfy a newer watcher ---
{
  const p2 = makePage();
  p2.beginGeneration(1); p2.finishGeneration(1);
  p2.beginGeneration(5);                    // in flight
  const v = terminalState(p2, 5);
  check('older receipt never satisfies a newer generation', !v.done);
}

// --- the real module must expose the same discipline ---
// generate() must clear the exposed receipt before awaiting anything.
import { readFileSync } from 'node:fs';
const mainSrc = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
// Strip comments before locating the first await — otherwise prose mentioning
// "await" is mistaken for code (this test's own earlier false positive).
const stripComments = (src) => src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '');
const genBody = stripComments(mainSrc.slice(mainSrc.indexOf('async function generate()')));
const clearIdx = genBody.search(/__kimodoLastReceipt\s*=\s*(null|undefined|\{)/);
const firstAwait = genBody.search(/\bawait\s/);
check('generate() supersedes the exposed receipt before its first await',
      clearIdx !== -1 && clearIdx < firstAwait,
      clearIdx === -1 ? 'no reset of __kimodoLastReceipt found' : `reset at ${clearIdx}, first await at ${firstAwait}`);
check('generate() stamps a generation id onto the receipt',
      /generationId/.test(mainSrc), 'no generationId in src/main.js');
check('harnesses compare a generation id',
      /generationId/.test(readFileSync(new URL('../tools/headless_smoke.mjs', import.meta.url), 'utf8')) &&
      /generationId/.test(readFileSync(new URL('../tools/filmstrip_smoke.mjs', import.meta.url), 'utf8')),
      'harnesses do not check generationId');

const failed = results.filter(r => !r.ok);
for (const r of results) console.log(`${r.ok ? 'PASS' : 'FAIL'}  ${r.name}${!r.ok && r.detail ? `   [${r.detail}]` : ''}`);
console.log(`\n${results.length - failed.length}/${results.length} passed`);
process.exit(failed.length ? 1 : 0);
