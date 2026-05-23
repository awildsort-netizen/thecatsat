// fixtures/validate.ts — quality gate for training.csv.
//
// What it checks:
//   1. required columns are present and ordered as declared;
//   2. snapshot_path resolves to a real file under fixtures/;
//   3. evidence_quote (when non-empty) appears verbatim in that snapshot;
//   4. no duplicate (url, title, observed_date) triples;
//   5. category / materiality_hint / source_type / expected_action are from
//      a small known vocabulary (the AF training basin, not free text).
//
// Run with: npx tsx parser_evolver/fixtures/validate.ts
//
// Exits non-zero on any failure so it can be wired into npm scripts later.

import { readFileSync, existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));

const REQUIRED_COLUMNS = [
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

type Column = (typeof REQUIRED_COLUMNS)[number];
type Row = Record<Column, string>;

const VOCAB: Readonly<Record<string, ReadonlySet<string>>> = {
  source_type: new Set(["status-page", "blog-index", "blog-post", "legal", "press-release", "marketing"]),
  category: new Set([
    "incident",
    "maintenance",
    "policy-update",
    "corporate-announcement",
    "product-update",
    "marketing",
    "not-found",
  ]),
  materiality_hint: new Set(["none", "low", "medium", "high"]),
  field_effect_hint: new Set(["none", "availability", "liquidity", "regulatory", "legal-terms"]),
  expected_action: new Set(["monitor", "archive", "flag-for-review", "ingest-as-fixture-only", "no-op"]),
};

// Minimal RFC-4180-ish CSV parser. Handles double-quoted fields, escaped
// quotes ("") inside them, and CRLF/LF line endings. No streaming — the
// fixture is small by design.
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
      // collapse CRLF to a single record boundary
      if (ch === "\r" && text[i + 1] === "\n") i += 2;
      else i += 1;
      // skip wholly empty trailing lines
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

function toRows(records: readonly string[][]): readonly Row[] {
  if (records.length === 0) throw new Error("CSV is empty");
  const header = records[0]!;
  const expected = [...REQUIRED_COLUMNS];
  if (header.length !== expected.length || expected.some((c, i) => header[i] !== c)) {
    throw new Error(`header mismatch.\n  expected: ${expected.join(",")}\n  got:      ${header.join(",")}`);
  }
  const rows: Row[] = [];
  for (let r = 1; r < records.length; r++) {
    const rec = records[r]!;
    if (rec.length !== expected.length) {
      throw new Error(`row ${r}: expected ${expected.length} columns, got ${rec.length}`);
    }
    const row = {} as Row;
    expected.forEach((c, i) => {
      row[c] = rec[i] ?? "";
    });
    rows.push(row);
  }
  return rows;
}

type Issue = { row: number; column?: Column; message: string };

function check(rows: readonly Row[]): readonly Issue[] {
  const issues: Issue[] = [];
  const seen = new Map<string, number>();

  rows.forEach((row, idx) => {
    const lineNo = idx + 2; // header is line 1

    // Vocabulary checks.
    for (const col of Object.keys(VOCAB) as Column[]) {
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

    // observed_date shape (ISO-8601 calendar date).
    if (!/^\d{4}-\d{2}-\d{2}$/.test(row.observed_date)) {
      issues.push({ row: lineNo, column: "observed_date", message: `not YYYY-MM-DD: '${row.observed_date}'` });
    }

    // url shape.
    if (!/^https?:\/\/\S+$/.test(row.url)) {
      issues.push({ row: lineNo, column: "url", message: `not an http(s) URL: '${row.url}'` });
    }

    // snapshot existence + evidence quote.
    const snap = resolve(HERE, row.snapshot_path);
    if (!existsSync(snap)) {
      issues.push({ row: lineNo, column: "snapshot_path", message: `file does not exist: ${row.snapshot_path}` });
    } else if (row.evidence_quote.trim() !== "") {
      const body = readFileSync(snap, "utf8");
      if (!body.includes(row.evidence_quote)) {
        issues.push({
          row: lineNo,
          column: "evidence_quote",
          message: `evidence not found verbatim in ${row.snapshot_path}: '${row.evidence_quote}'`,
        });
      }
    }

    // Duplicate (url, title, observed_date) triple.
    const key = `${row.url}␟${row.title}␟${row.observed_date}`;
    const prior = seen.get(key);
    if (prior !== undefined) {
      issues.push({ row: lineNo, message: `duplicate of row ${prior}: (url,title,observed_date) = ${key.replace(/␟/g, " | ")}` });
    } else {
      seen.set(key, lineNo);
    }
  });

  return issues;
}

function main(): void {
  const csvPath = resolve(HERE, "training.csv");
  if (!existsSync(csvPath)) {
    console.error(`missing fixture: ${csvPath}`);
    process.exit(2);
  }
  const text = readFileSync(csvPath, "utf8");
  const records = parseCsv(text);
  const rows = toRows(records);
  const issues = check(rows);

  console.log(`fixtures/training.csv: ${rows.length} rows, ${REQUIRED_COLUMNS.length} columns`);
  console.log(`  snapshots referenced: ${new Set(rows.map((r) => r.snapshot_path)).size}`);
  console.log(`  unique urls:          ${new Set(rows.map((r) => r.url)).size}`);

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

main();

// Exported for downstream reuse (e.g. a future demo that folds these rows
// into a CsvAF training run).
export { parseCsv, toRows, check, REQUIRED_COLUMNS, VOCAB };
export type { Row, Column, Issue };
