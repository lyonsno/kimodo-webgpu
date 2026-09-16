export const KIMODO_FRONTEND_TELEMETRY_SCHEMA = 'kimodo.frontend-runtime-evidence.v0';
export const KIMODO_ROUTE_ID = 'kimodo.text-to-motion.webgpu-local.v0';

function clone(value) {
  return value == null ? value : JSON.parse(JSON.stringify(value));
}

function frozenSnapshot(state) {
  const snapshot = clone(state);
  Object.freeze(snapshot.scheduler);
  Object.freeze(snapshot.progress);
  if (snapshot.route) Object.freeze(snapshot.route);
  if (snapshot.submission) Object.freeze(snapshot.submission);
  for (const stage of snapshot.timings) Object.freeze(stage);
  Object.freeze(snapshot.timings);
  return Object.freeze(snapshot);
}

/**
 * Page-local runtime evidence for the exact generation the operator is
 * watching. This does not invent GPU timing: live rows record frontend stage
 * and boundary observations, while queue counts and stage durations become
 * authoritative only when copied from the producer's terminal receipt.
 */
export function createFrontendTelemetry({
  generationId,
  numSteps,
  requestedMaxInFlightDuties,
  now = () => globalThis.performance?.now?.() ?? Date.now(),
} = {}) {
  if (!Number.isSafeInteger(generationId) || generationId <= 0) {
    throw new TypeError('generationId must be a positive safe integer');
  }
  if (!Number.isSafeInteger(numSteps) || numSteps <= 0) {
    throw new TypeError('numSteps must be a positive safe integer');
  }
  if (!Number.isSafeInteger(requestedMaxInFlightDuties) || requestedMaxInFlightDuties <= 0) {
    throw new TypeError('requestedMaxInFlightDuties must be a positive safe integer');
  }

  const startedAtMs = now();
  const state = {
    schema: KIMODO_FRONTEND_TELEMETRY_SCHEMA,
    source: 'live-page-generation',
    status: 'running',
    generationId,
    currentStage: 'admitted',
    elapsedMs: 0,
    route: {
      requestedRouteId: KIMODO_ROUTE_ID,
      effectiveRouteId: null,
      receiptStatus: 'in-progress',
    },
    scheduler: {
      mode: 'cooperative-foreground-boundary',
      requestedMaxInFlightDuties,
      expectedForegroundBoundaryCount: numSteps * 4,
      observedForegroundBoundaryCount: 0,
      lastBoundary: null,
      hostSubmissionCount: 0,
    },
    progress: { step: 0, numSteps, pct: 0 },
    submission: null,
    timings: [],
    failure: null,
  };

  const updateElapsed = () => {
    const elapsed = now() - startedAtMs;
    state.elapsedMs = Number.isFinite(elapsed) ? Math.max(0, Math.round(elapsed)) : null;
  };

  return Object.freeze({
    stage(name, event) {
      if (state.status !== 'running' || event !== 'start') return;
      state.currentStage = name;
      updateElapsed();
    },

    foreground(boundary = {}) {
      if (state.status !== 'running') return;
      state.scheduler.observedForegroundBoundaryCount += 1;
      state.scheduler.lastBoundary = {
        phase: boundary.phase ?? null,
        step: boundary.step ?? null,
        numSteps: boundary.numSteps ?? numSteps,
        pass: boundary.pass ?? null,
      };
      updateElapsed();
    },

    progress(progress = {}) {
      if (state.status !== 'running') return;
      state.progress = {
        step: progress.step ?? state.progress.step,
        numSteps: progress.numSteps ?? numSteps,
        pct: progress.pct ?? state.progress.pct,
      };
      updateElapsed();
    },

    succeed(receipt, submission = receipt?.metadata?.gpuSubmission ?? null) {
      if (state.status !== 'running') return;
      state.status = receipt?.status === 'real' ? 'succeeded' : 'invalid';
      state.currentStage = 'terminal';
      state.route = {
        requestedRouteId: receipt?.requestedRouteId ?? KIMODO_ROUTE_ID,
        effectiveRouteId: receipt?.effectiveRouteId ?? null,
        receiptStatus: receipt?.status ?? 'missing',
      };
      state.submission = clone(submission);
      state.scheduler.hostSubmissionCount = submission?.hostSubmissionCount ?? 0;
      state.timings = clone(receipt?.timings?.stages ?? []);
      state.elapsedMs = receipt?.timings?.totalMs ?? state.elapsedMs;
    },

    fail(error) {
      if (state.status !== 'running') return;
      state.status = error?.name === 'AbortError' || error?.phase === 'cancelled'
        ? 'cancelled'
        : 'failed';
      state.currentStage = error?.phase ?? 'exception';
      state.submission = clone(error?.gpuSubmission ?? null);
      state.scheduler.hostSubmissionCount = error?.gpuSubmission?.hostSubmissionCount ?? 0;
      state.failure = {
        name: error?.name ?? 'Error',
        message: error?.message ?? String(error),
        phase: error?.phase ?? 'exception',
      };
      updateElapsed();
    },

    snapshot() {
      return frozenSnapshot(state);
    },
  });
}

