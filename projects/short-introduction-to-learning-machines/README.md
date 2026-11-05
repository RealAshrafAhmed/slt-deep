# A Short Introduction to Learning Machines

**Research note.** The rigorous synthesis: five postulates, the learning machine
as a stochastic realization, the parameter space as a singular parametrisation,
and the connective-tissue table across physics, probability, statistics, systems
theory and machine learning.

## Audience

Theorists, physicists, mathematicians and applied statisticians. They know
measure-theoretic probability, some ergodic theory, information geometry or
singular learning theory, or classical/statistical mechanics; the note bridges
between those backgrounds and assumes none of them in full.

## Goal

To state, in one framework, the objects every learning problem involves (world,
instrument, record, symmetry, next-step kernel, realization, parametrisation) and
to show which quantities each field computes are properties of the kernel,
of the realization, or of the coordinates. Contribution: the assembly, with each
part cited to its source (`prior-art.md`). Later: Part 2 (actions: reports,
designs, control; equivalence of learning machines; invariants of the learning
dynamics).

## Standard of defensibility

- Every statement is a definition, a labelled assumption, a cited theorem (named,
  not reproduced), or an example.
- Every citation is checked against the source before it appears in prose
  (`verification.md`; `\verify{}` in the tex).
- Prior art is acknowledged per postulate; the contribution is claimed only for
  the combination.
- Rigour lives in the body (measurable spaces, filtrations, kernels, realization
  theory), not in an appendix.

## Files

| File | Role |
|---|---|
| `outline.md` | the outline to write from (was v8): postulates, §2–§7, provenance, decisions |
| `note.tex`, `note.pdf` | Part I draft of 2026-10-22; §1, §6, §7 reusable; §2–5 are the Foundations companion |
| `prior-art.md` | the search of 2026-10-25: ranked closest work, per-point coverage, what was not found |
| `verification.md` | every claim cited from memory, with status |
| `design-notes.md` | running-example detail (height data, HMM/accelerometer, BabyLM, 2×2 design) |
| `glossary.md` | term decisions; shared with the blog post, which applies them more strictly |
| `archive/` | outlines v1–v7 and LaTeX build files; history of the framing |

## Relation to the blog post

`../what-a-learning-machine-is-made-of/` is the plain-language note for new
researchers in applied ML. Same picture, same examples, same claims in spirit;
no theorem statements, no measure theory, citations only as pointers. Anything
the blog post asserts must be defensible by a statement in this note.
