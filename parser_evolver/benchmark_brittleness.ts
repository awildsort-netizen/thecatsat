// Benchmark for the motif-invariance brittleness metric.
// Run with: npx tsx parser_evolver/benchmark_brittleness.ts
//
// Two head-to-heads, each comparing the legacy solver (brittlenessWeight=0)
// against a brittleness-aware solver (brittlenessWeight > 0):
//
//   A) Same input, same operators. Measures whether the metric is doing
//      *something* — does it change which creature wins, and is the
//      winner's brittleness lower?
//
//   C) Train-then-stress. Pick the top creature on a "training" page.
//      Then evaluate that same gene-string on novelty pages: small drift
//      (date format flipped, blocks reordered) and larger drift (the
//      same page recomposed plus a less-structured PRNewswire-style
//      page). The brittleness-aware winner should survive novelty
//      better — that is the actual claim of the metric.
//
// The benchmark prints a markdown summary and writes a CSV row per
// configuration for downstream inspection.

import { companyUpdatesAF } from "./af.js";
import {
  brittlenessOf,
  reorderBlocks,
  swapDateFormat,
  dilateWhitespace,
  motifSimilarity,
  motifOf,
} from "./brittleness.js";
import { PRIMITIVES, makeEnforceSchema } from "./operators.js";
import { makeBeamSolver } from "./solver.js";
import type { ParseCandidate, ParseContext, RowHypothesis } from "./types.js";
import { writeFileSync } from "node:fs";

// ---------------------------------------------------------------------------
// Training and novelty contexts
// ---------------------------------------------------------------------------

const TRAINING = `
Blockchain.com Status

2026-04-12
Wallet sync latency elevated in EU region
We are investigating delays affecting balance refresh.
https://status.blockchain.com/incidents/wallet-eu-2026-04-12

2026-04-09
Scheduled maintenance for institutional API
Brief downtime expected between 02:00 and 03:00 UTC.
https://status.blockchain.com/maintenance/api-2026-04-09

April 1, 2026
Lightning withdrawals now generally available
After a long beta we are flipping the switch for all retail accounts.
https://www.blockchain.com/blog/posts/lightning-ga
`;

// Small-drift novelty: same shape, different ordering and date formats.
const SMALL_DRIFT = `
Blockchain.com Status

April 9, 2026
Scheduled maintenance for institutional API
Brief downtime expected between 02:00 and 03:00 UTC.
https://status.blockchain.com/maintenance/api-2026-04-09

2026-04-01
Lightning withdrawals now generally available
After a long beta we are flipping the switch for all retail accounts.
https://www.blockchain.com/blog/posts/lightning-ga

April 12, 2026
Wallet sync latency elevated in EU region
We are investigating delays affecting balance refresh.
https://status.blockchain.com/incidents/wallet-eu-2026-04-12
`;

// Larger drift: a PRNewswire-style block alongside Blockchain.com items.
// The structure is recognisable but headers and trailing chatter look
// different. A pattern-defending parser should still recover the iso
// dates and titles cleanly.
const LARGE_DRIFT = `
PRESS RELEASE

2026-05-03
Acme Capital Announces Acquisition of Beta Holdings
Acme Capital today announced it has entered into a definitive agreement.
https://www.prnewswire.com/news-releases/acme-acquires-beta-2026-05-03

2026-05-10
Beta Holdings Closes Series C Funding Round
The financing was led by an undisclosed lead investor.
https://www.prnewswire.com/news-releases/beta-series-c-2026-05-10

April 9, 2026
Scheduled maintenance for institutional API
Brief downtime expected between 02:00 and 03:00 UTC.
https://status.blockchain.com/maintenance/api-2026-04-09

Subscribe to updates if you want the firehose.
`;

const ctxOf = (text: string, url: string): ParseContext => ({
  url,
  rawText: text,
  normalizedText: text,
  sourceType: "status-page",
});

const trainingCtx = ctxOf(TRAINING, "https://status.blockchain.com/");
const smallDriftCtx = ctxOf(SMALL_DRIFT, "https://status.blockchain.com/?small-drift");
const largeDriftCtx = ctxOf(LARGE_DRIFT, "https://www.prnewswire.com/?large-drift");

const ops = [...PRIMITIVES, makeEnforceSchema(companyUpdatesAF)];

// ---------------------------------------------------------------------------
// A scoring helper that re-runs a candidate's gene-string on a fresh
// context, *with the legacy scoring rules* (no brittleness term). This
// is the honest evaluation: brittleness only earns its keep if the
// creatures it selects also do well under the original AF on novel data.
// ---------------------------------------------------------------------------

const baseSolverForEval = makeBeamSolver({ beam: 1, maxLen: 0, extensionTopK: 1 });

const evalCandidateOn = (
  gene: ParseCandidate,
  ctx: ParseContext,
): { rows: readonly RowHypothesis[]; legacyScore: number; brittleness: number } => {
  // We use seedGenes to lock the gene-string and maxLen=0 so the solver
  // does not extend; we just want the evaluation of this specific
  // gene-string in the new context.
  const sealed = makeBeamSolver({ beam: 1, maxLen: 0, extensionTopK: 1 });
  const cands = sealed.search(ctx, companyUpdatesAF, ops, [gene.genes]);
  const c = cands[0]!;
  const brittleness = brittlenessOf(c, ctx, ops).score;
  return { rows: c.rows, legacyScore: c.score, brittleness };
};

// Helper to render rows compactly.
const renderRows = (rows: readonly RowHypothesis[]): string =>
  rows
    .map((r) => {
      const f = r.fields;
      const date = f.date?.value ?? "—";
      const title = (f.title?.value ?? "—").slice(0, 40);
      return `${date} | ${title}`;
    })
    .join(" ; ");

// ---------------------------------------------------------------------------
// Benchmark A — same input, baseline vs. brittleness-aware
// ---------------------------------------------------------------------------

type RunResult = {
  readonly label: string;
  readonly weight: number;
  readonly topGenes: string;
  readonly trainingScore: number;
  readonly trainingBrittleness: number;
  readonly rowsOnTraining: number;
  readonly smallDriftScore: number;
  readonly smallDriftBrittleness: number;
  readonly smallDriftRows: number;
  readonly largeDriftScore: number;
  readonly largeDriftBrittleness: number;
  readonly largeDriftRows: number;
  readonly smallDriftMotifSim: number; // motif(training) vs motif(small)
  readonly largeDriftMotifSim: number; // motif(training) vs motif(large)
};

const runConfig = (label: string, weight: number): RunResult => {
  const solver = makeBeamSolver({
    beam: 8,
    maxLen: 6,
    extensionTopK: 4,
    brittlenessWeight: weight,
  });
  const cands = solver.search(trainingCtx, companyUpdatesAF, ops);
  const top = cands[0]!;
  const topGenes = top.genes.map((g) => g.operatorId).join(" >> ");

  const trainBrittle = brittlenessOf(top, trainingCtx, ops).score;
  const small = evalCandidateOn(top, smallDriftCtx);
  const large = evalCandidateOn(top, largeDriftCtx);

  const trainMotif = motifOf(top.rows);
  const smallMotif = motifOf(small.rows);
  const largeMotif = motifOf(large.rows);

  return {
    label,
    weight,
    topGenes,
    trainingScore: top.score,
    trainingBrittleness: trainBrittle,
    rowsOnTraining: top.rows.length,
    smallDriftScore: small.legacyScore,
    smallDriftBrittleness: small.brittleness,
    smallDriftRows: small.rows.length,
    largeDriftScore: large.legacyScore,
    largeDriftBrittleness: large.brittleness,
    largeDriftRows: large.rows.length,
    smallDriftMotifSim: motifSimilarity(trainMotif, smallMotif),
    largeDriftMotifSim: motifSimilarity(trainMotif, largeMotif),
  };
};

const WEIGHTS = [0, 1, 3, 7];
const results = WEIGHTS.map((w) => runConfig(w === 0 ? "baseline" : `brittle@${w}`, w));

// ---------------------------------------------------------------------------
// Print and persist
// ---------------------------------------------------------------------------

console.log("\n=== Benchmark A: baseline vs brittleness-aware on training input ===\n");
console.log(
  "| weight | top gene-string | rows | score | brittleness |\n" +
    "|--------|-----------------|------|-------|-------------|",
);
results.forEach((r) =>
  console.log(
    `| ${r.weight} | ${r.topGenes} | ${r.rowsOnTraining} | ${r.trainingScore.toFixed(3)} | ${r.trainingBrittleness.toFixed(4)} |`,
  ),
);

console.log("\n=== Benchmark C: novelty stress test (train on TRAINING, eval on drift) ===\n");
console.log(
  "| weight | small-drift rows | small-drift score | small motif-sim | large-drift rows | large-drift score | large motif-sim |\n" +
    "|--------|------------------|-------------------|-----------------|------------------|-------------------|-----------------|",
);
results.forEach((r) =>
  console.log(
    `| ${r.weight} | ${r.smallDriftRows} | ${r.smallDriftScore.toFixed(3)} | ${r.smallDriftMotifSim.toFixed(4)} | ${r.largeDriftRows} | ${r.largeDriftScore.toFixed(3)} | ${r.largeDriftMotifSim.toFixed(4)} |`,
  ),
);

// Headline numbers worth eyeballing first.
console.log("\n=== headline ===");
const baseline = results.find((r) => r.weight === 0)!;
const best = [...results].sort((a, b) => b.largeDriftScore - a.largeDriftScore)[0]!;
console.log(`baseline training score:           ${baseline.trainingScore.toFixed(3)}`);
console.log(`baseline training brittleness:     ${baseline.trainingBrittleness.toFixed(4)}`);
console.log(`baseline large-drift score:        ${baseline.largeDriftScore.toFixed(3)}`);
console.log(`best-on-drift weight:              ${best.weight}`);
console.log(`best-on-drift large-drift score:   ${best.largeDriftScore.toFixed(3)}`);
console.log(`best-on-drift training brittleness:${best.trainingBrittleness.toFixed(4)}`);

// CSV for downstream inspection (mirrors the existing repo convention).
const header =
  [
    "label",
    "weight",
    "top_genes",
    "training_score",
    "training_brittleness",
    "rows_on_training",
    "small_drift_score",
    "small_drift_brittleness",
    "small_drift_rows",
    "small_motif_sim",
    "large_drift_score",
    "large_drift_brittleness",
    "large_drift_rows",
    "large_motif_sim",
  ].join(",") + "\n";

const body = results
  .map((r) =>
    [
      r.label,
      r.weight,
      JSON.stringify(r.topGenes),
      r.trainingScore,
      r.trainingBrittleness,
      r.rowsOnTraining,
      r.smallDriftScore,
      r.smallDriftBrittleness,
      r.smallDriftRows,
      r.smallDriftMotifSim,
      r.largeDriftScore,
      r.largeDriftBrittleness,
      r.largeDriftRows,
      r.largeDriftMotifSim,
    ].join(","),
  )
  .join("\n");

writeFileSync("parser_evolver/benchmark_brittleness.csv", header + body + "\n");
console.log("\nwrote parser_evolver/benchmark_brittleness.csv");

// ---------------------------------------------------------------------------
// Benchmark D — contrastive selection on a fragility-prone search.
//
// The default operator set is already pattern-defending: every plausible
// gene-string converges to the same canonical creature, so the brittleness
// term has nothing to differentiate. To show the metric has teeth, we
// rank the *full beam* on a brittleness-fragile input — one where iso
// dates appear in some blocks and long dates in others. A creature that
// only ever emits iso dates will look fine on the training input but
// will fragment under swap_date_format. We then sort the candidates by
// brittleness and show the spread: do brittleness-aware picks rank
// differently from legacy-score picks?
// ---------------------------------------------------------------------------

const FRAGILE_TRAIN = `
Blockchain.com Status

2026-04-12
Wallet sync latency elevated in EU region
https://status.blockchain.com/incidents/wallet-eu-2026-04-12

April 9, 2026
Scheduled maintenance for institutional API
https://status.blockchain.com/maintenance/api-2026-04-09

2026-04-01
Lightning withdrawals now generally available
https://www.blockchain.com/blog/posts/lightning-ga
`;

const fragileCtx = ctxOf(FRAGILE_TRAIN, "https://status.blockchain.com/?fragile");

const legacySolver = makeBeamSolver({ beam: 16, maxLen: 6, extensionTopK: 5, brittlenessWeight: 0 });
const legacyBeam = legacySolver.search(fragileCtx, companyUpdatesAF, ops);

// Compute brittleness for each candidate in the legacy beam and show the
// top by each metric.
type RankedCand = {
  readonly genes: string;
  readonly score: number;
  readonly brittleness: number;
  readonly rows: number;
};

const ranked: RankedCand[] = legacyBeam.map((c) => ({
  genes: c.genes.map((g) => g.operatorId).join(" >> "),
  score: c.score,
  brittleness: brittlenessOf(c, fragileCtx, ops).score,
  rows: c.rows.length,
}));

const byScore = [...ranked].sort((a, b) => b.score - a.score);
const byBrittle = [...ranked].sort((a, b) => a.brittleness - b.brittleness || b.score - a.score);

console.log("\n=== Benchmark D: contrastive ranking on mixed-format input ===\n");
console.log("top 5 by legacy score:");
byScore.slice(0, 5).forEach((c, i) =>
  console.log(
    `  #${i} score=${c.score.toFixed(3)} brittleness=${c.brittleness.toFixed(4)} rows=${c.rows} genes=[${c.genes}]`,
  ),
);
console.log("\ntop 5 by brittleness (ties broken by score):");
byBrittle.slice(0, 5).forEach((c, i) =>
  console.log(
    `  #${i} brittleness=${c.brittleness.toFixed(4)} score=${c.score.toFixed(3)} rows=${c.rows} genes=[${c.genes}]`,
  ),
);

// Are the two rankings the same? If yes, brittleness picks the same
// creature legacy already picks (concordance). If no, brittleness is
// surfacing a different best candidate — the interesting case.
const concordant =
  byScore[0]?.genes === byBrittle[0]?.genes &&
  Math.abs((byScore[0]?.brittleness ?? 0) - (byBrittle[0]?.brittleness ?? 0)) < 1e-9;
console.log(`\nlegacy-best and brittleness-best agree: ${concordant}`);

// Brittleness spread across the beam: if everything has the same
// brittleness, the metric is uninformative on this search; if the spread
// is wide, brittleness is a real discriminator.
const brittleVals = ranked.map((r) => r.brittleness);
const minB = Math.min(...brittleVals);
const maxB = Math.max(...brittleVals);
console.log(
  `brittleness range across beam (n=${ranked.length}): min=${minB.toFixed(4)} max=${maxB.toFixed(4)} spread=${(maxB - minB).toFixed(4)}`,
);
