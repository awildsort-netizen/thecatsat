// Monitor pre-pass: snapshot -> parser_evolver -> structured candidate digest.
//
// The shape we want is small and honest:
//
//   training row + snapshot bytes
//       |
//       v
//   parser_evolver solver (CsvAF + primitives)
//       |
//       v
//   DigestRow: vocabulary-controlled fields + confidence + typed
//             hallucination pressure + trace_region_count
//
// Confidence is derived, not asserted: it folds together solver score,
// evidence-quote presence in the snapshot text, and the manifest's
// reachability note. When a snapshot is shell-only or 404, the pre-pass
// flips `expected_action` to `needs-rendered-fetch` / `flag-for-review`
// and drops confidence — better to admit the page didn't yield extraction
// than to manufacture an alert.
//
// The pre-pass intentionally does no network I/O and does not touch any
// scheduled monitor. It reads only the bounded fixture directory.

import { readFileSync, existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { companyUpdatesAF } from "../af.js";
import { PRIMITIVES, makeEnforceSchema } from "../operators.js";
import { makeBeamSolver } from "../solver.js";
import type {
  HallucinationKind,
  ParseCandidate,
  ParseContext,
} from "../types.js";
// Local CSV parser. We intentionally do NOT import from
// `../fixtures/validate.js` because that module runs the fixture
// validator at import time (top-level `main()`), which would
// contaminate the pre-pass's own stdout. Same RFC-4180-ish shape.
function parseCsv(text: string): readonly string[][] {
  const out: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let i = 0;
  let inQuotes = false;
  while (i < text.length) {
    const ch = text[i]!;
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          cell += '"';
          i += 2;
          continue;
        }
        inQuotes = false;
        i += 1;
        continue;
      }
      cell += ch;
      i += 1;
      continue;
    }
    if (ch === '"') {
      inQuotes = true;
      i += 1;
      continue;
    }
    if (ch === ",") {
      row.push(cell);
      cell = "";
      i += 1;
      continue;
    }
    if (ch === "\n" || ch === "\r") {
      row.push(cell);
      cell = "";
      if (ch === "\r" && text[i + 1] === "\n") i += 2;
      else i += 1;
      if (row.length === 1 && row[0] === "") {
        row = [];
        continue;
      }
      out.push(row);
      row = [];
      continue;
    }
    cell += ch;
    i += 1;
  }
  if (cell !== "" || row.length > 0) {
    row.push(cell);
    out.push(row);
  }
  return out;
}

export { parseCsv };
import {
  CATEGORY,
  EXPECTED_ACTION,
  FIELD_EFFECT,
  MATERIALITY,
  SOURCE_TYPE,
} from "./types.js";
import type {
  Category,
  DigestRow,
  ExpectedAction,
  FieldEffect,
  HallucinationKindCount,
  Materiality,
  SourceType,
} from "./types.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURE_ROOT = resolve(HERE, "..", "fixtures");

// ---------------------------------------------------------------------------
// Manifest + training CSV ingestion.
// ---------------------------------------------------------------------------

type ManifestEntry = {
  readonly source_id: string;
  readonly url: string;
  readonly snapshot_path: string;
  readonly http_status: string;
  readonly content_type: string;
  readonly reachable: boolean | "shell_only";
  readonly notes: string;
};

type Manifest = {
  readonly snapshots: readonly ManifestEntry[];
};

type TrainingRow = {
  readonly source_id: string;
  readonly source_type: string;
  readonly url: string;
  readonly observed_date: string;
  readonly title: string;
  readonly category: string;
  readonly materiality_hint: string;
  readonly field_effect_hint: string;
  readonly expected_action: string;
  readonly evidence_quote: string;
  readonly snapshot_path: string;
};

const TRAINING_COLUMNS = [
  "source_id",
  "source_type",
  "url",
  "observed_date",
  "title",
  "category",
  "materiality_hint",
  "field_effect_hint",
  "expected_action",
  "evidence_quote",
  "snapshot_path",
] as const;

function readManifest(path: string): Manifest {
  const text = readFileSync(path, "utf8");
  return JSON.parse(text) as Manifest;
}

function readTraining(path: string): readonly TrainingRow[] {
  const text = readFileSync(path, "utf8");
  const records = parseCsv(text);
  if (records.length === 0) throw new Error(`empty training csv: ${path}`);
  const header = records[0]!;
  const expected = [...TRAINING_COLUMNS];
  if (header.length !== expected.length || expected.some((c, i) => header[i] !== c)) {
    throw new Error(`training.csv header mismatch.\n  expected: ${expected.join(",")}\n  got:      ${header.join(",")}`);
  }
  return records.slice(1).map((rec) => {
    const row = {} as Record<string, string>;
    expected.forEach((c, i) => {
      row[c] = rec[i] ?? "";
    });
    return row as unknown as TrainingRow;
  });
}

// ---------------------------------------------------------------------------
// HTML -> visible text. Just enough to let the regex emitters fire.
// ---------------------------------------------------------------------------

const stripHtml = (html: string): string =>
  html
    // drop script/style blocks entirely
    .replace(/<(script|style)[^>]*>[\s\S]*?<\/\1>/gi, " ")
    // turn block-ish boundaries into newlines so the title regex has anchors
    .replace(/<\/(p|div|li|h[1-6]|section|article|header|footer|nav|main|aside|tr)>/gi, "\n")
    .replace(/<br\s*\/?>/gi, "\n")
    // strip remaining tags
    .replace(/<[^>]+>/g, " ")
    // common entities
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

// ---------------------------------------------------------------------------
// Solver wrapper.
// ---------------------------------------------------------------------------

const OPS = [...PRIMITIVES, makeEnforceSchema(companyUpdatesAF)];
const SOLVER = makeBeamSolver({ beam: 8, maxLen: 6, extensionTopK: 4 });

function runSolver(text: string, url: string, sourceType: string): ParseCandidate | undefined {
  const ctx: ParseContext = {
    url,
    rawText: text,
    normalizedText: text,
    sourceType,
  };
  const candidates = SOLVER.search(ctx, companyUpdatesAF, OPS);
  return candidates[0];
}

// ---------------------------------------------------------------------------
// Vocabulary coercion.
//
// Training labels are already validated against vocabulary by
// fixtures/validate.ts. The pre-pass re-asserts the type at the boundary
// so DigestRow's typed columns are not a lie. An unknown label is a
// programming error (training CSV drifted from vocabulary) — fail loud.
// ---------------------------------------------------------------------------

const intoVocab = <T extends string>(values: readonly T[], v: string, where: string): T => {
  if ((values as readonly string[]).includes(v)) return v as T;
  throw new Error(`unknown ${where} value '${v}'; not in vocabulary {${values.join(", ")}}`);
};

// ---------------------------------------------------------------------------
// Confidence + action policy.
//
// The pre-pass exposes the *honesty* of its extraction, not just a score.
// Three things move confidence:
//
//   1. evidence_quote is present in the snapshot bytes (binary signal);
//   2. solver score is positive (the AF found a stable parse at all);
//   3. reachability is healthy (manifest says true, not shell_only / false).
//
// `expected_action` is overridden when reachability is degraded:
//
//   - 404 / unreachable          -> flag-for-review
//   - shell_only (SPA shell)     -> needs-rendered-fetch
//
// Both signals come from the manifest the user already curated; the
// pre-pass does not reach the network.
// ---------------------------------------------------------------------------

type Reachability = boolean | "shell_only";

const reachabilityHealth = (r: Reachability): number => (r === true ? 1 : r === "shell_only" ? 0.3 : 0);

const computeConfidence = (
  evidenceFound: boolean,
  solverScore: number,
  reach: Reachability,
): number => {
  const evidenceWeight = evidenceFound ? 0.55 : 0;
  const solverWeight = solverScore > 0 ? Math.min(0.25, 0.05 * solverScore) : 0;
  const reachWeight = 0.2 * reachabilityHealth(reach);
  return Math.max(0, Math.min(1, evidenceWeight + solverWeight + reachWeight));
};

const overrideAction = (trained: ExpectedAction, reach: Reachability): ExpectedAction => {
  if (reach === false) return "flag-for-review";
  if (reach === "shell_only") return "needs-rendered-fetch";
  return trained;
};

// ---------------------------------------------------------------------------
// Hallucination pressure summary for a candidate.
// ---------------------------------------------------------------------------

const ALL_KINDS: readonly HallucinationKind[] = [
  "unsupported_cell",
  "misassigned_span",
  "field_role_confusion",
  "missing_emitter",
  "validator_rejection",
  "low_coverage_region",
  "overfit_pattern",
];

const kindCounts = (cand: ParseCandidate | undefined): HallucinationKindCount => {
  const summary = cand?.diagnostics.pressure?.summary;
  if (!summary) return Object.fromEntries(ALL_KINDS.map((k) => [k, 0]));
  // Trim to non-zero kinds to keep the digest small and human-readable.
  return Object.fromEntries(
    ALL_KINDS.filter((k) => (summary[k] ?? 0) > 0).map((k) => [k, summary[k] ?? 0]),
  );
};

// ---------------------------------------------------------------------------
// Build digest rows.
// ---------------------------------------------------------------------------

export type PrepassResult = {
  readonly rows: readonly DigestRow[];
  readonly perSnapshotCandidate: ReadonlyMap<string, ParseCandidate | undefined>;
};

const reachabilityOf = (manifest: Manifest, snapshotPath: string): Reachability => {
  const entry = manifest.snapshots.find((e) => e.snapshot_path === snapshotPath);
  return entry?.reachable ?? false;
};

export function runPrepass(opts?: { readonly fixtureRoot?: string }): PrepassResult {
  const root = opts?.fixtureRoot ?? FIXTURE_ROOT;
  const manifest = readManifest(resolve(root, "manifest.json"));
  const training = readTraining(resolve(root, "training.csv"));

  // Run solver once per unique snapshot, not once per training row.
  const candidateBySnapshot = new Map<string, ParseCandidate | undefined>();
  const textBySnapshot = new Map<string, string>();
  const uniqueSnapshots = Array.from(new Set(training.map((r) => r.snapshot_path)));
  uniqueSnapshots.forEach((snap) => {
    const file = resolve(root, snap);
    if (!existsSync(file)) {
      candidateBySnapshot.set(snap, undefined);
      textBySnapshot.set(snap, "");
      return;
    }
    const html = readFileSync(file, "utf8");
    const text = stripHtml(html);
    textBySnapshot.set(snap, text);
    const url = training.find((r) => r.snapshot_path === snap)?.url ?? "";
    const sourceType = training.find((r) => r.snapshot_path === snap)?.source_type ?? "";
    candidateBySnapshot.set(snap, runSolver(text, url, sourceType));
  });

  const rows: DigestRow[] = training.map((tr) => {
    const cand = candidateBySnapshot.get(tr.snapshot_path);
    const text = textBySnapshot.get(tr.snapshot_path) ?? "";
    const reach = reachabilityOf(manifest, tr.snapshot_path);
    const evidenceFound = tr.evidence_quote.trim() !== "" && text.includes(tr.evidence_quote);
    const solverScore = cand?.score ?? 0;
    const confidence = computeConfidence(evidenceFound, solverScore, reach);
    const sourceType = intoVocab(SOURCE_TYPE, tr.source_type, "source_type") as SourceType;
    const category = intoVocab(CATEGORY, tr.category, "category") as Category;
    const materiality = intoVocab(MATERIALITY, tr.materiality_hint, "materiality_hint") as Materiality;
    const fieldEffect = intoVocab(FIELD_EFFECT, tr.field_effect_hint, "field_effect_hint") as FieldEffect;
    const trainedAction = intoVocab(EXPECTED_ACTION, tr.expected_action, "expected_action") as ExpectedAction;
    const expectedAction = overrideAction(trainedAction, reach);
    return {
      source_id: tr.source_id,
      source_type: sourceType,
      url: tr.url,
      observed_date: tr.observed_date,
      title: tr.title,
      category,
      materiality_hint: materiality,
      field_effect_hint: fieldEffect,
      expected_action: expectedAction,
      confidence: Math.round(confidence * 1000) / 1000,
      hallucination_kinds: kindCounts(cand),
      trace_region_count: cand?.traces?.length ?? 0,
      evidence_quote: tr.evidence_quote,
      snapshot_path: tr.snapshot_path,
    };
  });

  return { rows, perSnapshotCandidate: candidateBySnapshot };
}

// ---------------------------------------------------------------------------
// Stable serialisation.
// ---------------------------------------------------------------------------

const csvEscape = (v: string): string =>
  /[",\r\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v;

const hallucinationJSON = (k: HallucinationKindCount): string => {
  // Sort keys for determinism so the checked-in digest doesn't churn.
  const keys = Object.keys(k).sort();
  const obj = Object.fromEntries(keys.map((key) => [key, k[key]]));
  return JSON.stringify(obj);
};

export function toCsv(rows: readonly DigestRow[]): string {
  const cols = [
    "source_id",
    "source_type",
    "url",
    "observed_date",
    "title",
    "category",
    "materiality_hint",
    "field_effect_hint",
    "expected_action",
    "confidence",
    "hallucination_kinds",
    "trace_region_count",
    "evidence_quote",
    "snapshot_path",
  ] as const;
  const lines = [cols.join(",")];
  for (const r of rows) {
    const cells: string[] = [
      r.source_id,
      r.source_type,
      r.url,
      r.observed_date,
      r.title,
      r.category,
      r.materiality_hint,
      r.field_effect_hint,
      r.expected_action,
      r.confidence.toFixed(3),
      hallucinationJSON(r.hallucination_kinds),
      String(r.trace_region_count),
      r.evidence_quote,
      r.snapshot_path,
    ];
    lines.push(cells.map(csvEscape).join(","));
  }
  return lines.join("\n") + "\n";
}

export function toJson(rows: readonly DigestRow[]): string {
  return JSON.stringify(rows, null, 2) + "\n";
}
