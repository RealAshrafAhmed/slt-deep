# What a Learning Machine Is Made Of

**Blog post** (one essay or four to five posts). The plain-language companion to
the research note in `../short-introduction-to-learning-machines/`.

## Audience

New researchers in applied machine learning: a master's or early PhD student, or
an engineer moving into research, with a bachelor's in computer science,
mathematics or physics. Comfortable with probability and basic ML; knows or will
accept undergraduate physics vocabulary (microstate, observable, ensemble,
macrostate) after a one-sentence definition. No measure theory, information
geometry or ergodic theory.

## Goal

Two things, in this order:

1. **Show that the two extremes are the same kind of object.** A language model
   and the Gaussian fitted in a second-year statistics course have the same
   parts: a world, an instrument, one long record, a rule for the next reading, a
   memory, a family of rules written with numbers, a score, a move, a check. What
   differs is which parts are large, hidden, or only approximable.
2. **Teach the statistician's stance toward an LLM.** How a statistician in a
   scientific experiment looks at a model of a phenomenon: what is the world,
   what does the instrument read, what symmetry lets one record stand in for
   many, what is the quantity of interest and is it learnable, what does the
   fitted object say about the phenomenon versus about the fit. The reader should
   leave able to ask those questions of an LLM.

Three uses of the picture close the post: what goes wrong when a part is misread,
how to build better machines by choosing parts deliberately, and how to translate
between fields that name the same part differently.

## Standard of defensibility

- Every claim is true as stated and is backed by a precise statement in the
  research note; nothing is asserted here that cannot be made rigorous there.
- No theorem statements, no proofs, no measure theory in the body. Named results
  appear in parentheses as pointers ("the ergodic theorem"). A one-page "careful
  version" appendix is allowed.
- Citations are a short reading list, not a bibliography.
- Writing rules in `glossary.md`: idea first, name second; one new term per
  paragraph; physics vocabulary defined once then used freely; other fields'
  vocabulary collected in the dictionary section.

## Files

| File | Role |
|---|---|
| `outline.md` | the outline to write from (was learning-framework v9): the two-machines opening, the purpose paragraph, §0–§7, the Gaussian ∣ LLM running table |
| `glossary.md` | the jargon audit and writing rules (copy of the research note's; this is where it is binding) |
| `archive-outline-spine-first.md` | the rejected v10 draft (table-first organisation); kept for its §4 "using the picture" material |

## Relation to the research note

Same picture, same three examples (heights, text, HMM/transformer), same
provenance. The research note carries the postulates, realization theory, the
singular-parametrisation geometry, the prior-art positioning and the full
cross-field table. This post carries the picture and the stance.
