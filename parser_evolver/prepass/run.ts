// CLI: write the pre-pass digest to stdout and to fixtures/digest.csv.
//
// Usage:
//   npx tsx parser_evolver/prepass/run.ts            # stdout only
//   npx tsx parser_evolver/prepass/run.ts --write    # also write digest.csv/.json
//
// The fixture digest is checked in as a deterministic example so reviewers
// can see what the pre-pass would hand to a downstream monitor without
// running it themselves.

import { writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { runPrepass, toCsv, toJson } from "./prepass.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURE_ROOT = resolve(HERE, "..", "fixtures");

function main(): void {
  const write = process.argv.includes("--write");
  const result = runPrepass();
  const csv = toCsv(result.rows);
  process.stdout.write(csv);

  // Brief stderr summary so it shows up in CI output without contaminating
  // the digest itself when piped.
  const total = result.rows.length;
  const flagged = result.rows.filter((r) => r.expected_action === "flag-for-review").length;
  const renderNeeded = result.rows.filter((r) => r.expected_action === "needs-rendered-fetch").length;
  const lowConf = result.rows.filter((r) => r.confidence < 0.4).length;
  process.stderr.write(
    `\nprepass: ${total} rows; flag-for-review=${flagged} needs-rendered-fetch=${renderNeeded} low-confidence(<0.4)=${lowConf}\n`,
  );

  if (write) {
    const csvPath = resolve(FIXTURE_ROOT, "digest.csv");
    const jsonPath = resolve(FIXTURE_ROOT, "digest.json");
    writeFileSync(csvPath, csv);
    writeFileSync(jsonPath, toJson(result.rows));
    process.stderr.write(`wrote ${csvPath}\nwrote ${jsonPath}\n`);
  }
}

main();
