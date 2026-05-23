// Test the monitor pre-pass against the bounded fixtures.
//
// These assertions encode the contract we want the pre-pass to honour for
// any downstream monitor: digest rows are evidence-backed, vocabulary-
// controlled, IPO-shaped degraded sources are not silently fabricated, and
// trace regions exist where extraction succeeded.

import { runPrepass, toCsv, parseCsv } from "./prepass.js";
import { check, toRows } from "./validate-digest.js";
import {
  CATEGORY,
  EXPECTED_ACTION,
  FIELD_EFFECT,
  MATERIALITY,
  SOURCE_TYPE,
} from "./types.js";

const assertions: { name: string; ok: boolean; detail?: string }[] = [];
const assert = (name: string, ok: boolean, detail?: string) =>
  assertions.push({ name, ok, detail });

const result = runPrepass();
const { rows } = result;

assert("pre-pass produces at least one row", rows.length > 0, `rows=${rows.length}`);

// Vocabulary.
assert(
  "every source_type is in vocabulary",
  rows.every((r) => (SOURCE_TYPE as readonly string[]).includes(r.source_type)),
);
assert(
  "every category is in vocabulary",
  rows.every((r) => (CATEGORY as readonly string[]).includes(r.category)),
);
assert(
  "every materiality_hint is in vocabulary",
  rows.every((r) => (MATERIALITY as readonly string[]).includes(r.materiality_hint)),
);
assert(
  "every field_effect_hint is in vocabulary",
  rows.every((r) => (FIELD_EFFECT as readonly string[]).includes(r.field_effect_hint)),
);
assert(
  "every expected_action is in vocabulary (including escalation actions)",
  rows.every((r) => (EXPECTED_ACTION as readonly string[]).includes(r.expected_action)),
);

// Confidence in [0,1].
assert(
  "confidence is in [0,1]",
  rows.every((r) => r.confidence >= 0 && r.confidence <= 1),
);

// trace_region_count non-negative integer.
assert(
  "trace_region_count is a non-negative integer",
  rows.every((r) => Number.isInteger(r.trace_region_count) && r.trace_region_count >= 0),
);

// IPO degraded-source contract.
const ipoBlog = rows.find((r) => r.source_id === "blog-ipo-announce");
assert(
  "IPO blog row is present",
  ipoBlog !== undefined,
);
assert(
  "IPO blog row is escalated to needs-rendered-fetch (shell_only snapshot)",
  ipoBlog?.expected_action === "needs-rendered-fetch",
  `got=${ipoBlog?.expected_action}`,
);
assert(
  "IPO blog row does NOT claim high confidence (snapshot is shell-only)",
  (ipoBlog?.confidence ?? 1) < 0.85,
  `conf=${ipoBlog?.confidence}`,
);

const prnewswire = rows.find((r) => r.source_id === "prnewswire-ipo");
assert(
  "PRNewswire 404 row is present",
  prnewswire !== undefined,
);
assert(
  "PRNewswire 404 row is escalated to flag-for-review",
  prnewswire?.expected_action === "flag-for-review",
  `got=${prnewswire?.expected_action}`,
);

// High-materiality rows obey the source/escalation rule.
const highMat = rows.filter((r) => r.materiality_hint === "high");
const officialOrEscalated = highMat.every(
  (r) =>
    r.source_type === "legal" ||
    r.source_type === "status-page" ||
    r.source_type === "press-release" ||
    r.expected_action === "flag-for-review" ||
    r.expected_action === "needs-rendered-fetch",
);
assert(
  "every high-materiality row is on an official source OR escalated",
  officialOrEscalated,
  `high=${highMat.length}`,
);

// Evidence backing — the validator already enforces this, but assert
// directly here too so the test is honest about what it's checking.
assert(
  "every digest row carries a non-empty evidence_quote",
  rows.every((r) => r.evidence_quote.trim() !== ""),
);

// Confidence ordering: at least one row that is reachable+evidence-found
// should outscore the shell_only IPO row. Otherwise the pre-pass is not
// distinguishing degraded sources from healthy ones.
const reachableRows = rows.filter(
  (r) => r.source_id !== "blog-ipo-announce" && r.source_id !== "prnewswire-ipo",
);
const maxReachableConfidence = Math.max(...reachableRows.map((r) => r.confidence), 0);
assert(
  "best reachable confidence is strictly higher than shell_only IPO confidence",
  maxReachableConfidence > (ipoBlog?.confidence ?? 1),
  `reachable_max=${maxReachableConfidence} ipo=${ipoBlog?.confidence}`,
);

// CSV round-trip: emitted CSV parses back to the same number of rows
// and the validator finds no issues against the live fixtures.
const csv = toCsv(rows);
const parsed = parseCsv(csv);
assert(
  "emitted CSV parses back to expected row count",
  parsed.length - 1 === rows.length,
  `parsed_data_rows=${parsed.length - 1} expected=${rows.length}`,
);
const digestRows = toRows(parsed);
const issues = check(digestRows);
assert(
  "validator finds no issues in pre-pass output",
  issues.length === 0,
  issues.length === 0 ? undefined : `first=${JSON.stringify(issues[0])}`,
);

// At least one row should have non-zero trace_region_count, proving the
// pre-pass is actually exercising parser_evolver's trace machinery on
// the fixtures rather than skating past it.
assert(
  "at least one row carries trace regions",
  rows.some((r) => r.trace_region_count > 0),
);

const failed = assertions.filter((a) => !a.ok);
assertions.forEach((a) =>
  console.log(`${a.ok ? "ok" : "FAIL"}  ${a.name}${a.detail ? `  [${a.detail}]` : ""}`),
);
console.log(`\n${assertions.length - failed.length}/${assertions.length} passed`);
if (failed.length > 0) (globalThis as { process?: { exit: (n: number) => void } }).process?.exit(1);
