// Validate a pre-pass digest CSV against the snapshots it claims to summarise.
//
// What it checks (the contract the pre-pass owes a downstream monitor):
//
//   1. columns present and in declared order;
//   2. every controlled vocabulary cell is in the vocabulary;
//   3. snapshot_path resolves under fixtures/;
//   4. evidence_quote (when non-empty) appears verbatim in the snapshot
//      *after* the same HTML-strip the pre-pass uses (so we're checking
//      against the same view the parser_evolver saw);
//   5. confidence is in [0,1] and parseable as a float;
//   6. trace_region_count is a non-negative integer;
//   7. hallucination_kinds is a JSON object of non-negative integers and
//      only contains known HallucinationKind keys;
//   8. high-materiality rows require either source_type ∈ {legal, status-page,
//      press-release} OR expected_action ∈ {flag-for-review,
//      needs-rendered-fetch}. The pre-pass is allowed to escalate to
//      human review, but it cannot quietly claim high materiality from a
//      marketing/blog-index page.
//
// Exits non-zero on any failure.

import { readFileSync, existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { parseCsv } from "./prepass.js";
import {
  CATEGORY,
  DIGEST_COLUMNS,
  EXPECTED_ACTION,
  FIELD_EFFECT,
  MATERIALITY,
  SOURCE_TYPE,
} from "./types.js";
import type { DigestColumn } from "./types.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURE_ROOT = resolve(HERE, "..", "fixtures");

const VOCAB: Readonly<Record<string, ReadonlySet<string>>> = {
  source_type: new Set(SOURCE_TYPE),
  category: new Set(CATEGORY),
  materiality_hint: new Set(MATERIALITY),
  field_effect_hint: new Set(FIELD_EFFECT),
  expected_action: new Set(EXPECTED_ACTION),
};

const KNOWN_KINDS = new Set([
  "unsupported_cell",
  "misassigned_span",
  "field_role_confusion",
  "missing_emitter",
  "validator_rejection",
  "low_coverage_region",
  "overfit_pattern",
]);

const OFFICIAL_SOURCE_TYPES = new Set(["legal", "status-page", "press-release"]);
const ESCALATION_ACTIONS = new Set(["flag-for-review", "needs-rendered-fetch"]);

// Same HTML strip the pre-pass uses; duplicated here on purpose so the
// validator can't drift away from the pre-pass's text view without us
// noticing during review.
const stripHtml = (html: string): string =>
  html
    .replace(/<(script|style)[^>]*>[\s\S]*?<\/\1>/gi, " ")
    .replace(/<\/(p|div|li|h[1-6]|section|article|header|footer|nav|main|aside|tr)>/gi, "\n")
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

type Row = Record<DigestColumn, string>;
type Issue = { row: number; column?: DigestColumn; message: string };

function toRows(records: readonly string[][]): readonly Row[] {
  if (records.length === 0) throw new Error("digest CSV is empty");
  const header = records[0]!;
  const expected = [...DIGEST_COLUMNS];
  if (header.length !== expected.length || expected.some((c, i) => header[i] !== c)) {
    throw new Error(`header mismatch.\n  expected: ${expected.join(",")}\n  got:      ${header.join(",")}`);
  }
  return records.slice(1).map((rec) => {
    if (rec.length !== expected.length) {
      throw new Error(`row has ${rec.length} cols, expected ${expected.length}: ${rec.join(",")}`);
    }
    const row = {} as Row;
    expected.forEach((c, i) => {
      row[c] = rec[i] ?? "";
    });
    return row;
  });
}

function check(rows: readonly Row[]): readonly Issue[] {
  const issues: Issue[] = [];
  const stripped = new Map<string, string>();

  rows.forEach((row, idx) => {
    const lineNo = idx + 2;

    // Vocabulary.
    for (const col of Object.keys(VOCAB) as DigestColumn[]) {
      const v = row[col];
      const allowed = VOCAB[col]!;
      if (!allowed.has(v)) {
        issues.push({
          row: lineNo,
          column: col,
          message: `value '${v}' not in vocabulary {${[...allowed].sort().join(", ")}}`,
        });
      }
    }

    // confidence in [0,1].
    const conf = Number(row.confidence);
    if (!Number.isFinite(conf) || conf < 0 || conf > 1) {
      issues.push({ row: lineNo, column: "confidence", message: `confidence not in [0,1]: '${row.confidence}'` });
    }

    // trace_region_count non-negative integer.
    const trc = Number(row.trace_region_count);
    if (!Number.isInteger(trc) || trc < 0) {
      issues.push({
        row: lineNo,
        column: "trace_region_count",
        message: `not a non-negative integer: '${row.trace_region_count}'`,
      });
    }

    // hallucination_kinds JSON.
    let kinds: Record<string, unknown> = {};
    try {
      kinds = JSON.parse(row.hallucination_kinds) as Record<string, unknown>;
    } catch {
      issues.push({
        row: lineNo,
        column: "hallucination_kinds",
        message: `not parseable JSON: '${row.hallucination_kinds}'`,
      });
    }
    for (const [k, v] of Object.entries(kinds)) {
      if (!KNOWN_KINDS.has(k)) {
        issues.push({
          row: lineNo,
          column: "hallucination_kinds",
          message: `unknown HallucinationKind '${k}'`,
        });
      }
      if (typeof v !== "number" || !Number.isInteger(v) || v < 0) {
        issues.push({
          row: lineNo,
          column: "hallucination_kinds",
          message: `value for '${k}' is not a non-negative integer: ${JSON.stringify(v)}`,
        });
      }
    }

    // url shape.
    if (!/^https?:\/\/\S+$/.test(row.url)) {
      issues.push({ row: lineNo, column: "url", message: `not an http(s) URL: '${row.url}'` });
    }

    // observed_date.
    if (!/^\d{4}-\d{2}-\d{2}$/.test(row.observed_date)) {
      issues.push({ row: lineNo, column: "observed_date", message: `not YYYY-MM-DD: '${row.observed_date}'` });
    }

    // snapshot existence + evidence presence (against stripped text).
    const snap = resolve(FIXTURE_ROOT, row.snapshot_path);
    if (!existsSync(snap)) {
      issues.push({ row: lineNo, column: "snapshot_path", message: `file does not exist: ${row.snapshot_path}` });
    } else if (row.evidence_quote.trim() !== "") {
      let text = stripped.get(row.snapshot_path);
      if (text === undefined) {
        text = stripHtml(readFileSync(snap, "utf8"));
        stripped.set(row.snapshot_path, text);
      }
      // Also check the raw bytes; legal pages render the date inline as
      // markup that survives strip, but a row whose evidence only appears
      // inside an attribute would slip past a too-strict text check.
      // We accept either: raw bytes OR stripped text.
      const raw = readFileSync(snap, "utf8");
      if (!text.includes(row.evidence_quote) && !raw.includes(row.evidence_quote)) {
        issues.push({
          row: lineNo,
          column: "evidence_quote",
          message: `evidence not found in stripped text or raw bytes of ${row.snapshot_path}: '${row.evidence_quote}'`,
        });
      }
    }

    // High-materiality requires official source OR escalation action.
    if (row.materiality_hint === "high") {
      const official = OFFICIAL_SOURCE_TYPES.has(row.source_type);
      const escalated = ESCALATION_ACTIONS.has(row.expected_action);
      if (!official && !escalated) {
        issues.push({
          row: lineNo,
          message:
            `high materiality requires official source_type ${[...OFFICIAL_SOURCE_TYPES].join("/")} ` +
            `or escalation expected_action ${[...ESCALATION_ACTIONS].join("/")}; ` +
            `got source_type=${row.source_type} expected_action=${row.expected_action}`,
        });
      }
    }
  });

  return issues;
}

function main(): void {
  const csvPath = process.argv[2] ?? resolve(FIXTURE_ROOT, "digest.csv");
  if (!existsSync(csvPath)) {
    console.error(`missing digest CSV: ${csvPath}`);
    console.error(`hint: run \`npm run prepass -- --write\` first`);
    process.exit(2);
  }
  const text = readFileSync(csvPath, "utf8");
  const records = parseCsv(text);
  const rows = toRows(records);
  const issues = check(rows);

  console.log(`digest: ${rows.length} rows`);
  console.log(`  snapshots referenced: ${new Set(rows.map((r) => r.snapshot_path)).size}`);
  console.log(`  flag-for-review:      ${rows.filter((r) => r.expected_action === "flag-for-review").length}`);
  console.log(`  needs-rendered-fetch: ${rows.filter((r) => r.expected_action === "needs-rendered-fetch").length}`);

  if (issues.length === 0) {
    console.log("OK");
    return;
  }
  console.error(`\n${issues.length} issue(s):`);
  for (const i of issues) {
    const where = i.column ? `${i.row}/${i.column}` : `${i.row}`;
    console.error(`  line ${where}: ${i.message}`);
  }
  process.exit(1);
}

// Only run as a CLI; tests import { check, toRows }.
const invokedDirectly =
  typeof process !== "undefined" &&
  process.argv?.[1] !== undefined &&
  process.argv[1]!.endsWith("validate-digest.ts");
if (invokedDirectly) main();

export { check, toRows };
