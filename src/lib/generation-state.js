/**
 * Generation lifecycle: one source of truth for what a generation's evidence
 * state is and when a watcher may treat it as terminal.
 *
 * The app (src/main.js), both smoke harnesses, and the tests all consume THIS
 * module. Before it existed the same logic lived in four divergent copies; the
 * copies drifted (one harness did not recognize a real failure status and
 * turned it into a timeout), and a test exercised its own local model rather
 * than the shipped code. Divergence is now impossible rather than tested-for.
 *
 * Receipt statuses: 'in-progress' -> exactly one of 'real' | 'invalid' | 'failed'.
 * Every started generation whose awaits SETTLE reaches a terminal status;
 * ensureTerminalReceipt is the structural backstop that closes any settled
 * path that forgot.
 *
 * Known limitation: the embedding fetch carries no deadline, so an endpoint
 * that accepts the connection and never completes the response leaves the
 * generation in-progress indefinitely — the backstop runs in a finally block
 * that an unsettled await never reaches. Watchers must own their timeout and
 * treat it as failure.
 */

/** Non-authoritative marker published at generation start, before any await. */
export function inProgressReceipt(generationId) {
  return {
    status: 'in-progress',
    generationId,
    createdAt: new Date().toISOString(),
  };
}

/** Terminal failure evidence for one generation: phase names the boundary that failed. */
export function failureReceipt(generationId, phase, reason) {
  return {
    status: 'failed',
    generationId,
    phase,
    reason: String(reason ?? ''),
    createdAt: new Date().toISOString(),
  };
}

/**
 * Backstop: if `receipt` is still in-progress for the generation that just
 * settled, convert it to a terminal failure. Any other receipt (terminal, or
 * belonging to a different generation) passes through untouched.
 */
export function ensureTerminalReceipt(receipt, generationId, phase = 'incomplete') {
  if (receipt && receipt.generationId === generationId && receipt.status === 'in-progress') {
    return failureReceipt(generationId, phase, 'generation settled without a terminal receipt');
  }
  return receipt;
}

/**
 * The single classifier a watcher uses to decide whether the generation it is
 * watching has terminally succeeded or failed.
 *
 * Rules a watcher may rely on:
 * - Evidence belonging to any other generation NEVER terminates the wait.
 * - A null/undefined expectedId is never "universally fresh": the watcher must
 *   know which generation it is watching, or it waits (and its own timeout
 *   fails loud) rather than blessing stale evidence.
 * - 'real' additionally requires the rendered canvas to be present.
 * - An unknown status terminates as failure rather than hanging.
 */
export function classifyGenerationState({ receipt, expectedId, canvasPresent }) {
  if (!receipt) return { done: false, reason: 'no-receipt' };
  if (expectedId == null) return { done: false, reason: 'no-expected-id' };
  if (receipt.generationId !== expectedId) return { done: false, reason: 'stale-receipt' };
  switch (receipt.status) {
    case 'in-progress':
      return { done: false, reason: 'in-progress' };
    case 'failed':
      return { done: true, ok: false, phase: receipt.phase, detail: receipt.reason };
    case 'invalid':
      return { done: true, ok: false, detail: receipt.fallbackReason ?? 'invalid output' };
    case 'real':
      return canvasPresent
        ? { done: true, ok: true }
        : { done: false, reason: 'awaiting-canvas' };
    default:
      return { done: true, ok: false, detail: `unknown receipt status "${receipt.status}"` };
  }
}
