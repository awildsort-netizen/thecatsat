// CsvAF — Company updates / news table attractor basin.
//
// Climate definition: required columns (date, title, url), a row constraint
// that every cell must point to a real source span, and a scoring function
// that rewards coverage and source diversity while heavily penalising
// hallucinated cells.
//
// Hallucinations are typed: see `hallucinations(rows)` below. The AF reads
// the typed kinds and turns them into penalty weight; the solver also
// surfaces them in `ParseDiagnostics.pressure` so downstream operators can
// react in kind.

import type { ColumnSpec, CsvAF, FieldHypothesis, Hallucination, HallucinationKind, RowHypothesis } from "./types.js";

const isDateish = (v: string): boolean =>
  /^\d{4}-\d{2}-\d{2}$/.test(v) || /^[A-Z][a-z]+\s+\d{1,2},\s+\d{4}$/.test(v);

const isUrl = (v: string): boolean => /^https?:\/\/\S+$/.test(v);

const isTitleish = (v: string): boolean =>
  v.length >= 8 && v.length <= 160 && !isDateish(v) && !isUrl(v);

const VALIDATORS: Readonly<Record<string, (v: string) => boolean>> = {
  date: isDateish,
  title: isTitleish,
  url: isUrl,
};

export const COLUMNS: readonly ColumnSpec[] = [
  { name: "date", required: true, validators: [VALIDATORS.date!] },
  { name: "title", required: true, validators: [VALIDATORS.title!] },
  { name: "url", required: false, validators: [VALIDATORS.url!] },
];

// A cell is sourced when its span is non-empty and bounded. The text-length
// check would need the source text in scope; we approximate by requiring
// strictly positive span width — emitters that fail this are immediately
// flagged `unsupported_cell`.
const hasNonEmptySpan = (fh: FieldHypothesis): boolean => fh.span[0] >= 0 && fh.span[1] > fh.span[0];

const allCellsSourced = (row: RowHypothesis): boolean =>
  Object.values(row.fields).every(hasNonEmptySpan);

const requiredsPresent = (row: RowHypothesis): boolean =>
  COLUMNS.filter((c) => c.required).every((c) => row.fields[c.name] !== undefined);

const cellsValidate = (row: RowHypothesis): boolean =>
  COLUMNS.every((c) => {
    const fh = row.fields[c.name];
    return fh === undefined ? !c.required : (c.validators ?? []).every((v) => v(fh.value));
  });

const rowConstraints = [allCellsSourced, requiredsPresent, cellsValidate] as const;

// ---------------------------------------------------------------------------
// Typed hallucination detection.
//
// Each kind is detected by a pure predicate over the field set. Coverage of
// kinds is deliberately small: the seed gives downstream code a clean
// switchable surface, not a finished taxonomy.
// ---------------------------------------------------------------------------

const cellsOf = (rows: readonly RowHypothesis[]): readonly FieldHypothesis[] =>
  rows.flatMap((r) => Object.values(r.fields));

const unsupportedCells = (rows: readonly RowHypothesis[]): readonly Hallucination[] =>
  cellsOf(rows)
    .filter((c) => !hasNonEmptySpan(c))
    .map<Hallucination>((c) => ({
      kind: "unsupported_cell",
      operator: c.operator,
      field: c.field,
      span: c.span,
      weight: 5,
      note: "cell carries no usable source span",
    }));

const validatorRejections = (rows: readonly RowHypothesis[]): readonly Hallucination[] =>
  cellsOf(rows)
    .filter((c) => VALIDATORS[c.field] && !VALIDATORS[c.field]!(c.value))
    .map<Hallucination>((c) => ({
      kind: "validator_rejection",
      operator: c.operator,
      field: c.field,
      span: c.span,
      weight: 2,
      note: `value ${JSON.stringify(c.value.slice(0, 30))} failed validator for ${c.field}`,
    }));

const fieldRoleConfusions = (rows: readonly RowHypothesis[]): readonly Hallucination[] =>
  cellsOf(rows)
    .flatMap((c) =>
      Object.entries(VALIDATORS)
        .filter(([otherField, v]) => otherField !== c.field && v(c.value) && !(VALIDATORS[c.field] ?? (() => true))(c.value))
        .map<Hallucination>(([otherField]) => ({
          kind: "field_role_confusion",
          operator: c.operator,
          field: c.field,
          span: c.span,
          weight: 3,
          note: `value validates as ${otherField} but was emitted as ${c.field}`,
        })),
    );

const missingEmitters = (rows: readonly RowHypothesis[]): readonly Hallucination[] => {
  // Only meaningful once *some* row exists: an empty run is uninformative,
  // not lying, and we don't want to make non-empty creatures look worse.
  if (rows.length === 0) return [];
  const provided = new Set(cellsOf(rows).map((c) => c.field));
  return COLUMNS.filter((c) => c.required && !provided.has(c.name)).map<Hallucination>((c) => ({
    kind: "missing_emitter",
    field: c.name,
    weight: 2,
    note: `no operator contributed to required column ${c.name}`,
  }));
};

const lowCoverageRegion = (rows: readonly RowHypothesis[]): readonly Hallucination[] =>
  rows.length === 0
    ? [{ kind: "low_coverage_region", weight: 0.5, note: "no rows assembled" } as const]
    : [];

const overfitPattern = (rows: readonly RowHypothesis[]): readonly Hallucination[] => {
  const perField: Record<string, number> = cellsOf(rows).reduce<Record<string, number>>(
    (acc, c) => ({ ...acc, [c.field]: (acc[c.field] ?? 0) + 1 }),
    {},
  );
  // If any field emits more than 3× the number of rows, the pattern is
  // claiming more than the table can absorb.
  return Object.entries(perField)
    .filter(([, n]) => rows.length > 0 && n > 3 * rows.length)
    .map<Hallucination>(([field, n]) => ({
      kind: "overfit_pattern",
      field,
      weight: 1,
      note: `${field} emitted ${n} cells across ${rows.length} rows`,
    }));
};

const hallucinations = (rows: readonly RowHypothesis[]): readonly Hallucination[] => [
  ...unsupportedCells(rows),
  ...validatorRejections(rows),
  ...fieldRoleConfusions(rows),
  ...missingEmitters(rows),
  ...lowCoverageRegion(rows),
  ...overfitPattern(rows),
];

// ---------------------------------------------------------------------------
// Scoring.
// ---------------------------------------------------------------------------

const coverage = (row: RowHypothesis): number =>
  COLUMNS.filter((c) => row.fields[c.name] !== undefined).length / COLUMNS.length;

const scoreRow = (row: RowHypothesis): number =>
  coverage(row) * row.score - (cellsValidate(row) ? 0 : 1);

const distinctTitles = (rows: readonly RowHypothesis[]): number =>
  new Set(rows.map((r) => r.fields.title?.value).filter(Boolean)).size;

const scoreRun = (rows: readonly RowHypothesis[]): number => {
  const kept = rows.filter((r) => rowConstraints.every((c) => c(r)));
  const sum = kept.reduce((s, r) => s + scoreRow(r), 0);
  const diversity = 0.25 * Math.max(0, distinctTitles(kept) - 1);
  const penalty = hallucinations(rows).reduce((s, h) => s + h.weight, 0);
  return sum + diversity - penalty;
};

export const companyUpdatesAF: CsvAF = {
  columns: COLUMNS,
  rowConstraints,
  scoreRow,
  scoreRun,
  hallucinations,
};

// Re-exported so tests/diagnostics can summarise pressure without duplication.
export const summarisePressure = (
  rows: readonly RowHypothesis[],
): Record<HallucinationKind, number> =>
  hallucinations(rows).reduce<Record<string, number>>(
    (acc, h) => ({ ...acc, [h.kind]: (acc[h.kind] ?? 0) + 1 }),
    {
      unsupported_cell: 0,
      misassigned_span: 0,
      field_role_confusion: 0,
      missing_emitter: 0,
      validator_rejection: 0,
      low_coverage_region: 0,
      overfit_pattern: 0,
    },
  ) as Record<HallucinationKind, number>;
