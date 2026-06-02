# AGENTS.md - IFLS Mental Health x Temperature


## Project context

**Working title:** "Heat, Stress, and Mental Health in Indonesia: Evidence from IFLS"

**Core thesis:** Ambient temperature on or near the survey date worsens
self-reported mental-health outcomes (CES-D depressive symptoms, sleep,
subjective well-being), and the effect is amplified among respondents already
exposed to baseline stressors. Heat does not create distress in a vacuum; it
tips already-stressed people over the edge.

## Structure

`code`: contains all code.

- `data`: scripts that pre-process raw IFLS and environmental data into
  analysis-ready datasets. See `code/data/DATA.md` before editing the data
  pipeline.
- `analysis`: regression, table, and figure scripts that consume the canonical
  analysis input.
    - Note: always the final analysis dataset `30_analysis_table_input.parquet` as input unless excplicitly told not to or unless the user's request 
    requires data at a different level
    - `exploratory`: prototype or one-off code such as `_tmp_*.py` and `_tmp_*.R`.
    - `lib`: shared utilities where present. Reuse compatible helpers from
    - `tables`: final tables to be included in the paper. 
    - `figures`: final tables to be included in the paper. 
        - Use plotnine to make figures


`data`: contains raw and generated data. The detailed raw/extracted/generated
layout and large-data rules live in `code/data/DATA.md`.

`resources`: relevant documents and notes.

- `papers`: cited paper PDFs, also tracked in pyzotero where possible.
- `summary.md`, `mechanisms.md`: mirror Simon's literature and mechanism tables;
  reuse his coverage where possible and add Indonesia-specific entries.

Each directory may contain a `README.md` with task-specific guidance.

## Data pipeline

All data-specific programming guidance has moved to `code/data/DATA.md`.
Read that file before changing raw IFLS handling, generated data contracts,
GEE/environmental pulls, numbered pipeline steps, Pandera schemas, or
data-script helper patterns.

## pyzotero

Reuse Simon's collection: same paper set, same mental-health-temperature lit.

- `uv run pyzotero` to manage citations and resources.
- Collection ID: `S9JPSTQK` ("Mental Health and Temperature").
- `uv run pyzotero search --collection S9JPSTQK` lists all papers.
- `uv run pyzotero search --collection S9JPSTQK -q "INSERT_QUERY"` finds a
  specific paper.
- `uv run pyzotero children "PAPER_KEY" --json` fetches attachments; PDFs are
  available via `enclosure.href: file://`.

When adding IFLS-specific or Indonesia-specific papers, add them to the same
collection so both projects benefit. See `../simon/resources/summary.md` and
`mechanisms.md` for existing coverage; do not duplicate it.

## Guidelines

- Use `uv` to run code and add dependencies (`uv add`, `uv run python ...`).
- Keep edits minimal and script-first unless the user asks for packaging or
  tooling expansion.
- Code should made to read top-down. If the code requires decomposition for clarity then functions names should be clear
  and such that the final main funciton can be read like a set of instructions telling the reader what is being done
- If you add dev tooling or workflows, document the exact commands in the
  relevant repo guide. Ask the user before doing so
- **Identification framing:** within-individual variation across waves with
  date-by-community fixed effects is the cleanest cut, but the headline spec is
  closer to Mullins & White 2019: community fixed effects, month-of-year fixed
  effects, and year fixed effects, identifying off daily weather wiggles around
  the interview. State the assumption explicitly in code comments where the
  regression lives.

## Anti-patterns

- Do not create new pyzotero collections; extend `S9JPSTQK`.
- Do not duplicate data-pipeline rules in this file. Keep them in
  `code/data/DATA.md`.
