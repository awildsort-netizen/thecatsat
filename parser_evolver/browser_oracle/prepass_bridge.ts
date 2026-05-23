// Bridge: TraceDistillation -> prepass hint.
//
// The prepass already escalates SPA-shell sources to
// `needs-rendered-fetch` (see ../prepass/prepass.ts). This bridge takes
// a TraceDistillation produced by an *external* browser oracle for one
// of those escalated rows and turns it into a `PrepassHint`: a
// structured object a downstream monitor could read instead of
// re-launching a browser. We do not mutate the digest CSV; the prepass
// is the cheap structural step and stays the source of truth there.
//
// The bridge intentionally honours remembered absence: a distillation
// that did not cover required evidence becomes a hint that still says
// `needs-rendered-fetch`, but explains *why* the most recent oracle
// trace was not enough — so the monitor doesn't loop on the same trace.

import type { DigestRow } from "../prepass/types.js";
import type { ProposedStaticOperator, RememberedAbsence, TraceDistillation } from "./types.js";

export type PrepassHintAction =
  | "use-static-operator"
  | "keep-needs-rendered-fetch"
  | "remembered-absence";

export type PrepassHint = {
  readonly source_id: string;
  readonly url: string;
  readonly action: PrepassHintAction;
  readonly canRetireBrowser: boolean;
  readonly proposedOperators: readonly ProposedStaticOperator[];
  readonly rememberedAbsences: readonly RememberedAbsence[];
  readonly distillationConfidence: number;
  // What the prepass *would* have done before the hint. Helps a future
  // monitor reason about deltas without re-reading the digest CSV.
  readonly priorExpectedAction: DigestRow["expected_action"];
  readonly note: string;
};

export function hintFromDistillation(
  row: DigestRow,
  distillation: TraceDistillation,
): PrepassHint {
  const hasOperators = distillation.proposedOperators.length > 0;
  const action: PrepassHintAction = distillation.canRetireBrowser
    ? "use-static-operator"
    : hasOperators
      ? "keep-needs-rendered-fetch"
      : "remembered-absence";

  const note = (() => {
    if (action === "use-static-operator") {
      return "Trace produced every required evidence field from a minimal request set; static operator proposal is ready to lift.";
    }
    if (action === "keep-needs-rendered-fetch") {
      return "Trace produced partial evidence; keep browser fallback armed until the gaps close.";
    }
    return "Trace did not cover required evidence; remembered absence recorded so the next tick does not summon a browser hoping for a different result.";
  })();

  return {
    source_id: row.source_id,
    url: row.url,
    action,
    canRetireBrowser: distillation.canRetireBrowser,
    proposedOperators: distillation.proposedOperators,
    rememberedAbsences: distillation.rememberedAbsences,
    distillationConfidence: distillation.confidence,
    priorExpectedAction: row.expected_action,
    note,
  };
}
