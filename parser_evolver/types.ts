// Parser-evolver core types.
//
// CSV is an attractor basin, not an output format. A parser-creature is a
// short gene-string of operators whose decompression under the CsvAF climate
// either folds the page into a stable table or doesn't. Field hypotheses
// always carry a source span: cells that cannot point back to text are
// hallucinations.
//
// Hallucinations are not just a scalar penalty here — they are typed,
// bounded artifacts (Hallucination + TraceRegion) that downstream operators
// can inspect and react to. The hook is deliberately minimal so it can grow
// into a Riordan-style flow regression later without redesigning the seed.

export type Span = readonly [start: number, end: number];

export type FieldHypothesis = {
  readonly field: string;
  readonly value: string;
  readonly span: Span;
  readonly operator: string;
  readonly confidence: number;
  readonly evidence?: string;
  // Optional pointer into the run's trace regions; lets a cell carry more
  // than just a source span when an upstream operator wants to annotate
  // *where in the dataflow* the cell came from.
  readonly traceRegionId?: string;
};

export type RowHypothesis = {
  readonly fields: Readonly<Record<string, FieldHypothesis>>;
  readonly score: number;
};

export type ParseContext = {
  readonly url: string;
  readonly rawText: string;
  readonly normalizedText: string;
  readonly sourceType?: string;
};

// ---------------------------------------------------------------------------
// Hallucination annotations as first-class typed artifacts.
//
// A hallucination is any place where the parser-creature is making a claim
// the source can't back up. Keeping these typed (not a single scalar) lets
// the AF and downstream operators distinguish "cell with no span" from
// "cell whose role contradicts another cell's role" from "validator
// rejection". A `FailurePressure` reducer can later count persistent kinds
// per run and propose new emitters/validators — that hook is intentionally
// type-only here.
// ---------------------------------------------------------------------------

export type HallucinationKind =
  | "unsupported_cell"        // span empty or points outside source
  | "misassigned_span"        // span belongs to a different field's region
  | "field_role_confusion"    // value validates as field A but was emitted as field B
  | "missing_emitter"         // required column has no emitter contributing
  | "validator_rejection"     // value present but failed column validator
  | "low_coverage_region"     // a TraceRegion that produced no kept cells
  | "overfit_pattern";        // pattern matched far more often than columns can absorb

export type Hallucination = {
  readonly kind: HallucinationKind;
  readonly operator?: string;
  readonly field?: string;
  readonly span?: Span;
  readonly note?: string;
  readonly weight: number;    // suggested AF penalty contribution
};

// A TraceRegion is a labelled bounded region of the dataflow — a span in
// source text plus a label, an originating operator, and the channel it
// flowed through. It is the minimum data hook for later flow-regression
// passes; current operators emit them lazily, and the AF reads them as
// optional evidence alongside `FieldHypothesis.span`.
export type TraceRegion = {
  readonly id: string;
  readonly label: string;
  readonly span: Span;
  readonly channel: string;
  readonly operator: string;
};

// FailurePressure is the type-level hook for "persistent hallucinations
// propose new operators". Today the solver only reads `summary` for
// diagnostics; tomorrow a mutation operator can use `propose` to draft an
// extension to the operator ecology. Kept tiny on purpose.
export type FailurePressure = {
  readonly summary: Readonly<Record<HallucinationKind, number>>;
  readonly propose?: () => readonly string[];
};

// ---------------------------------------------------------------------------
// Operators: tendencies advertised by signature.
// ---------------------------------------------------------------------------

export type OperatorSignature = {
  readonly needs: readonly string[];
  readonly provides: readonly string[];
  // Symbolic embedding tokens drawn from name/comment/file/purpose. Used by
  // the solver's embedding similarity to discover relatives without
  // hand-wired conditionals.
  readonly tokens: readonly string[];
};

export type ParseOperator = {
  readonly id: string;
  readonly cost: number;
  readonly signature: OperatorSignature;
  readonly run: (ctx: ParseContext, input: unknown) => unknown;
};

export type ColumnSpec = {
  readonly name: string;
  readonly required?: boolean;
  readonly validators?: readonly ((value: string) => boolean)[];
};

// CsvAF — the attractor basin. The AF doesn't pick "the right parse"; it
// punishes unstable decompressions and lets the stable ones keep their
// footing. scoreRow/scoreRun are the climate's expense accounting; the
// optional hallucinations(rows) hook reports typed pressure for diagnostics
// and for solver penalty.
export type CsvAF = {
  readonly columns: readonly ColumnSpec[];
  readonly rowConstraints: readonly ((row: RowHypothesis) => boolean)[];
  readonly scoreRow: (row: RowHypothesis) => number;
  readonly scoreRun: (rows: readonly RowHypothesis[]) => number;
  readonly hallucinations: (rows: readonly RowHypothesis[]) => readonly Hallucination[];
};

// Bytecode-flavoured gene. Typed instruction with optional params — not an
// opaque closure. Distributions over genes are the natural search frontier.
export type Gene = {
  readonly operatorId: string;
  readonly params?: Readonly<Record<string, unknown>>;
};

export type GeneString = readonly Gene[];

export type ParseDiagnostics = {
  readonly coverage: number;
  readonly complexity: number;
  readonly hallucinationRisk: number;
  readonly stability?: number;
  // First-class typed pressure, summarised per kind.
  readonly pressure?: FailurePressure;
};

export type ParseCandidate = {
  readonly genes: GeneString;
  readonly rows: readonly RowHypothesis[];
  readonly score: number;
  readonly diagnostics: ParseDiagnostics;
  // TraceRegions accumulated during decompression. Empty for now in most
  // creatures; lets future operators inspect dataflow regions cheaply.
  readonly traces?: readonly TraceRegion[];
};

export type Solver = {
  readonly search: (
    ctx: ParseContext,
    af: CsvAF,
    operators: readonly ParseOperator[],
    seedGenes?: readonly GeneString[],
  ) => readonly ParseCandidate[];
};

// Shader-style design hook (not implemented). A heap-of-rows could later
// lower to a kernel(row, columnIndex) -> cell pipeline, evaluated in a
// uniform climate (the AF). Keeping rows as flat readonly arrays of
// hypotheses preserves that future.
export type RowKernel = (row: RowHypothesis, columnIndex: number) => FieldHypothesis | undefined;
