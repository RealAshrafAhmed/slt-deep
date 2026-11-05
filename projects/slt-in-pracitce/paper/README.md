# SLT in Practice — Paper

Working drafts for *Singular Learning Theory for Applied Deep Learning*. The
`prep/` documents capture the goals, needs, and analysis that scaffold the paper
before the main manuscript is written.

## Structure

```
slt-in-pracitce/paper/
├── prep/
│   ├── goals.tex       # Primary goal and objectives
│   ├── analysis.tex    # Analysis / annotations
│   └── needs.tex       # Requirements (with priority legend)
├── tasks.mise.toml     # Build task definitions
└── README.md
```

## Building the PDFs

All tasks are run from the **monorepo root** (`slt-deep/`). They wrap
`scripts/publish.sh`, which invokes the LaTeX toolchain and writes each PDF next
to its `.tex` source in `prep/`.

```bash
# Compile every prep document
mise run sltp-paper:all

# Compile a single document
mise run sltp-paper:pdf-compile-goals
mise run sltp-paper:pdf-compile-analysis
mise run sltp-paper:pdf-compile-needs

# Remove build artifacts (.aux, .log, .out, ...) from prep/
mise run sltp-paper:clean-prep
```

### Available tasks

| Task | Description |
|------|-------------|
| `sltp-paper:all` | Compile all prep PDFs (alias for `pdf-compile-all-prep`) |
| `sltp-paper:pdf-compile-all-prep` | Compile `goals`, `analysis`, and `needs` |
| `sltp-paper:pdf-compile-goals` | Compile `prep/goals.tex` → `prep/goals.pdf` |
| `sltp-paper:pdf-compile-analysis` | Compile `prep/analysis.tex` → `prep/analysis.pdf` |
| `sltp-paper:pdf-compile-needs` | Compile `prep/needs.tex` → `prep/needs.pdf` |
| `sltp-paper:clean-prep` | Clean LaTeX build artifacts from `prep/` |

List the tasks at any time with `mise tasks ls | grep sltp`.

## Compiling manually

The tasks are thin wrappers; you can call the publish script directly from the
monorepo root (omit the `.tex` extension):

```bash
./scripts/publish.sh compile slt-in-pracitce/paper/prep/goals
./scripts/publish.sh clean slt-in-pracitce/paper/prep
```
