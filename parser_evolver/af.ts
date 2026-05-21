// CsvAF — Company updates / news table attractor basin.
//
// Climate definition: required columns (date, title, url), a row constraint
// that every cell must point to a real source span, and a scoring function
// that rewards coverage and stable assembly while heavily penalising
// hallucinated cells (cells with no span or invalid spans).

import type { CsvAF, ColumnSpec, FieldHypothesis, RowHypothesis } from "./types.js";

const isDateish = (v: string): boolean =>
  /^\d{4}-\d{2}-\d{2}$/.test(v) || /^[A-Z][a-z]+\s+\d{1,2},\s+\d{4}$/.test(v);

const isUrl = (v: string): boolean => /^https?:\/\/\S+$/.test(v);

const isTitleish = (v: string): boolean => v.length >= 8 && v.length <= 160 && !/^https?:\/\//.test(v);

export const COLUMNS: readonly ColumnSpec[] = [
  { name: "date", required: true, validators: [isDateish] },
  { name: "title", required: true, validators: [isTitleish] },
  { name: "url", required: false, validators: [isUrl] },
];

const hasValidSpan = (fh: FieldHypothesis, textLength: number): boolean =>
  fh.span[0] >= 0 && fh.span[1] > fh.span[0] && fh.span[1] <= textLength;

const allCellsSourced = (row: RowHypothesis): boolean =>
  Object.values(row.fields).every((f) => hasValidSpan(f, Number.MAX_SAFE_INTEGER));

const requiredsPresent = (row: RowHypothesis): boolean =>
  COLUMNS.filter((c) => c.required).every((c) => row.fields[c.name] !== undefined);

const cellsValidate = (row: RowHypothesis): boolean =>
  COLUMNS.every((c) => {
    const fh = row.fields[c.name];
    if (fh === undefined) return !c.required;
    return (c.validators ?? []).every((v) => v(fh.value));
  });

const rowConstraints = [allCellsSourced, requiredsPresent, cellsValidate] as const;

const coverage = (row: RowHypothesis): number =>
  COLUMNS.filter((c) => row.fields[c.name] !== undefined).length / COLUMNS.length;

const hallucinationPenalty = (row: RowHypothesis): number =>
  Object.values(row.fields).filter((f) => f.span[1] <= f.span[0]).length;

const scoreRow = (row: RowHypothesis): number =>
  coverage(row) * row.score - 3 * hallucinationPenalty(row) - (cellsValidate(row) ? 0 : 1);

const distinctTitles = (rows: readonly RowHypothesis[]): number =>
  new Set(rows.map((r) => r.fields.title?.value).filter(Boolean)).size;

const scoreRun = (rows: readonly RowHypothesis[]): number => {
  const kept = rows.filter((r) => rowConstraints.every((c) => c(r)));
  const sum = kept.reduce((s, r) => s + scoreRow(r), 0);
  // Diversity bonus: distinct titles past the first reward source diversity.
  return sum + 0.25 * Math.max(0, distinctTitles(kept) - 1);
};

export const companyUpdatesAF: CsvAF = {
  columns: COLUMNS,
  rowConstraints,
  scoreRow,
  scoreRun,
};
