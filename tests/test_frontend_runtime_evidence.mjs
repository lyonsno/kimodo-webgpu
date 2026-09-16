/**
 * Frontend runtime-evidence contract.
 *
 * The operator smoke must make the producer's actual scheduling seam and
 * terminal queue/timing evidence visible on the same page that generated the
 * motion. A progress percentage alone is not evidence that bounded GPU duties
 * or foreground opportunities were exercised.
 */

import { readFileSync } from 'node:fs';
import { createFrontendTelemetry, KIMODO_FRONTEND_TELEMETRY_SCHEMA } from '../src/lib/frontend-telemetry.js';

let passed = 0;
let failed = 0;

function check(name, condition, detail = '') {
  if (condition) {
    passed++;
    console.log(`  PASS ${name}`);
  } else {
    failed++;
    console.error(`  FAIL ${name}${detail ? `: ${detail}` : ''}`);
  }
}

const indexSrc = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const mainSrc = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');

check('the page has a human-visible runtime evidence panel',
  indexSrc.includes('id="runtime-evidence"'));
check('the panel exposes scheduler state',
  indexSrc.includes('id="scheduler-state"'));
check('the panel exposes foreground-boundary state',
  indexSrc.includes('id="foreground-state"'));
check('the panel exposes bounded GPU queue state',
  indexSrc.includes('id="gpu-queue-state"'));
check('the panel exposes stage timing evidence',
  indexSrc.includes('id="timing-state"'));
check('runtime evidence announces meaningful changes without stealing focus',
  /id="runtime-evidence"[^>]*aria-live="polite"/.test(indexSrc));

check('the standalone smoke exercises the producer foreground boundary',
  /foregroundOpportunity\s*:/.test(mainSrc));
check('the live evidence snapshot is available to browser witnesses',
  mainSrc.includes('window.__kimodoFrontendTelemetry'));
check('terminal success renders the producer submission report',
  /renderTerminalTelemetry\s*\(\s*receipt/.test(mainSrc));
check('terminal failure renders failure-side submission evidence',
  /renderFailureTelemetry\s*\(\s*err/.test(mainSrc));

let clock = 100;
const telemetry = createFrontendTelemetry({
  generationId: 7,
  numSteps: 2,
  requestedMaxInFlightDuties: 2,
  now: () => clock,
});
telemetry.stage('ddim-sampling', 'start');
for (const pass of ['cond-root', 'cond-body', 'uncond-root', 'uncond-body']) {
  clock += 5;
  telemetry.foreground({ phase: 'ddim-sampling', step: 1, numSteps: 2, pass });
}
telemetry.progress({ step: 1, numSteps: 2, pct: 50 });
let live = telemetry.snapshot();
check('live telemetry is generation-bound and records every foreground boundary',
  live.schema === KIMODO_FRONTEND_TELEMETRY_SCHEMA
    && live.generationId === 7
    && live.scheduler.observedForegroundBoundaryCount === 4
    && live.scheduler.expectedForegroundBoundaryCount === 8
    && live.scheduler.lastBoundary.pass === 'uncond-body',
  JSON.stringify(live));
check('live telemetry does not pretend a terminal queue report already exists',
  live.status === 'running' && live.submission === null && live.route.receiptStatus === 'in-progress',
  JSON.stringify(live));

const submission = {
  status: 'drained',
  maxInFlightDuties: 2,
  maxObservedInFlightDuties: 2,
  submittedDutyCount: 8,
  completedDutyCount: 8,
  failedDutyCount: 0,
  inFlightDutyCount: 0,
  hostSubmissionCount: 0,
};
const oneStepSubmission = {
  ...submission,
  submittedDutyCount: 4,
  completedDutyCount: 4,
};
telemetry.succeed({
  status: 'real',
  generationId: 7,
  requestedRouteId: 'kimodo.text-to-motion.webgpu-local.v0',
  effectiveRouteId: 'kimodo.text-to-motion.webgpu-local.v0',
  timings: { totalMs: 321, stages: [{ name: 'ddim-sampling', durationMs: 300 }] },
  metadata: { gpuSubmission: submission },
}, submission);
const terminal = telemetry.snapshot();
check('terminal telemetry uses the producer report instead of inferred queue counts',
  terminal.status === 'succeeded'
    && terminal.submission.completedDutyCount === 8
    && terminal.submission.inFlightDutyCount === 0
    && terminal.scheduler.hostSubmissionCount === 0,
  JSON.stringify(terminal));
check('terminal telemetry preserves effective route and stage timing identity',
  terminal.route.effectiveRouteId === 'kimodo.text-to-motion.webgpu-local.v0'
    && terminal.route.receiptStatus === 'real'
    && terminal.timings[0].name === 'ddim-sampling'
    && terminal.timings[0].durationMs === 300,
  JSON.stringify(terminal));

function terminalProbe(generationId, receiptGenerationId, gpuSubmission) {
  const probe = createFrontendTelemetry({
    generationId,
    numSteps: 1,
    requestedMaxInFlightDuties: 2,
    now: () => 0,
  });
  probe.succeed({
    status: 'real',
    generationId: receiptGenerationId,
    requestedRouteId: 'kimodo.text-to-motion.webgpu-local.v0',
    effectiveRouteId: 'kimodo.text-to-motion.webgpu-local.v0',
    timings: { totalMs: 10, stages: [] },
    metadata: gpuSubmission === undefined ? {} : { gpuSubmission },
  });
  return probe.snapshot();
}

const wrongGeneration = terminalProbe(7, 8, submission);
check('a terminal receipt from another generation cannot publish success',
  wrongGeneration.status !== 'succeeded'
    && wrongGeneration.failure?.code === 'receipt-generation-mismatch'
    && wrongGeneration.failure?.expectedGenerationId === 7
    && wrongGeneration.failure?.actualGenerationId === 8,
  JSON.stringify(wrongGeneration));

const missingSubmission = terminalProbe(7, 7, undefined);
check('a real receipt with no bounded-submission report cannot publish success',
  missingSubmission.status !== 'succeeded'
    && missingSubmission.failure?.code === 'submission-report-missing',
  JSON.stringify(missingSubmission));

const activeSubmission = terminalProbe(7, 7, {
  ...submission,
  status: 'active',
  completedDutyCount: 7,
  inFlightDutyCount: 1,
});
check('a nonterminal bounded-submission report cannot publish success',
  activeSubmission.status !== 'succeeded'
    && activeSubmission.failure?.code === 'submission-report-nonterminal',
  JSON.stringify(activeSubmission));

const failedSubmission = terminalProbe(7, 7, {
  ...submission,
  status: 'failed',
  completedDutyCount: 7,
  failedDutyCount: 1,
});
check('a failed bounded-submission report cannot publish success',
  failedSubmission.status !== 'succeeded'
    && failedSubmission.failure?.code === 'submission-report-nonterminal',
  JSON.stringify(failedSubmission));

const peakExceedsLifetime = terminalProbe(7, 7, {
  ...oneStepSubmission,
  submittedDutyCount: 1,
  completedDutyCount: 1,
  maxObservedInFlightDuties: 2,
});
check('an observed peak above lifetime submitted duties cannot publish success',
  peakExceedsLifetime.status !== 'succeeded'
    && peakExceedsLifetime.failure?.code === 'submission-report-invalid',
  JSON.stringify(peakExceedsLifetime));

const zeroDutySubmission = terminalProbe(7, 7, {
  ...oneStepSubmission,
  submittedDutyCount: 0,
  completedDutyCount: 0,
  maxObservedInFlightDuties: 0,
});
check('a zero-duty report for a positive-step generation cannot publish success',
  zeroDutySubmission.status !== 'succeeded'
    && zeroDutySubmission.failure?.code === 'submission-report-invalid',
  JSON.stringify(zeroDutySubmission));

const wrongDutyCount = terminalProbe(7, 7, submission);
check('a drained report with the wrong producer-duty count cannot publish success',
  wrongDutyCount.status !== 'succeeded'
    && wrongDutyCount.failure?.code === 'submission-report-invalid',
  JSON.stringify(wrongDutyCount));

const wrongCapacity = terminalProbe(7, 7, {
  ...oneStepSubmission,
  maxInFlightDuties: 3,
});
check('a drained report from a different queue capacity cannot publish success',
  wrongCapacity.status !== 'succeeded'
    && wrongCapacity.failure?.code === 'submission-report-invalid',
  JSON.stringify(wrongCapacity));

const sameGenerationDrained = terminalProbe(7, 7, oneStepSubmission);
check('a same-generation real receipt with a drained report remains successful',
  sameGenerationDrained.status === 'succeeded'
    && sameGenerationDrained.failure === null
    && sameGenerationDrained.submission?.status === 'drained',
  JSON.stringify(sameGenerationDrained));

const failedTelemetry = createFrontendTelemetry({
  generationId: 8,
  numSteps: 1,
  requestedMaxInFlightDuties: 2,
  now: () => 0,
});
failedTelemetry.fail(Object.assign(new Error('queue rejected'), {
  phase: 'ddim-sampling',
  gpuSubmission: { status: 'failed', submittedDutyCount: 1, completedDutyCount: 0, failedDutyCount: 1, inFlightDutyCount: 0 },
}));
const failedSnapshot = failedTelemetry.snapshot();
check('failure telemetry keeps the failure phase and actual queue report',
  failedSnapshot.status === 'failed'
    && failedSnapshot.failure.phase === 'ddim-sampling'
    && failedSnapshot.submission.failedDutyCount === 1,
  JSON.stringify(failedSnapshot));

console.log(`\n${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
