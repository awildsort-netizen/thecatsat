// Parser-evolver core types.
//
// CSV is an attractor basin, not an output format. A parser-creature is a
// short gene-string of operators whose decompression under the CsvAF climate
// either folds the page into a stable table or doesn't. Field hypotheses
// always carry a source span: cells that cannot point back to text are
// hallucinations.

export type Span = readonly [start: number, end: number];

export type FieldHypothesis = {
  readonly field: string;
  readonly value: string;
  readonly span: Span;
  readonly operator: string;
  readonly confidence: number;
  readonly evidence?: string;
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

// An operator advertises its needs/provides through its signature. Composition
// is discovered from these tokens; we don't hand-wire pipelines.
export type OperatorSignature = {
  readonly needs: readonly string[];
  readonly provides: readonly string[];
  // Symbolic embedding tokens drawn from name/comment/file/purpose. Used for
  // similarity-based polymorphism (see embedding.ts).
  readonly tokens: readonly string[];
};

export type ParseOperator = {
  readonly id: string;
  readonly cost: number;
  readonly signature: OperatorSignature;
  // Operators are tendencies, not pure functions: same input, same climate
  // gives the same output here, but the signature is what allows the solver
  // to discover compositions without conditionals.
  readonly run: (ctx: ParseContext, input: unknown) => unknown;
};

export type ColumnSpec = {
  readonly name: string;
  readonly required?: boolean;
  readonly validators?: readonly ((value: string) => boolean)[];
};

// CsvAF — the attractor basin. The AF doesn't pick "the right parse"; it
// punishes unstable decompressions and lets the stable ones keep their
// footing. scoreRow / scoreRun are the climate's expense accounting.
export type CsvAF = {
  readonly columns: readonly ColumnSpec[];
  readonly rowConstraints: readonly ((row: RowHypothesis) => boolean)[];
  readonly scoreRow: (row: RowHypothesis) => number;
  readonly scoreRun: (rows: readonly RowHypothesis[]) => number;
};

// Bytecode-flavored gene. Distribution-friendly: a Gene is a typed
// instruction with optional params, not an opaque closure. A GeneString is a
// sequence of such instructions; weighted distributions over genes live in
// solver.ts as the search frontier.
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
};

export type ParseCandidate = {
  readonly genes: GeneString;
  readonly rows: readonly RowHypothesis[];
  readonly score: number;
  readonly diagnostics: ParseDiagnostics;
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
// uniform climate (the AF). Keeping the row store as a flat readonly array
// of hypotheses preserves that future.
export type RowKernel = (row: RowHypothesis, columnIndex: number) => FieldHypothesis | undefined;
