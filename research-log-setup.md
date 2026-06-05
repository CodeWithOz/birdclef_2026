# Research Log — Setup Instructions for AI Agents

## Purpose

This document instructs an AI agent on how to create and maintain a structured research log
(`research/log.md`) inside a competition repository. The log tracks experiments, decisions,
and lessons in a way that future AI sessions can load and use to avoid repeating past work.

---

## File location

Create the file at `research/log.md` inside the repository root. If the `research/` directory
does not exist, create it.

---

## File structure

The file has three sections, always in this order.

### Section 1: Experiments

A markdown table with these columns:

| Date | Model / variation | Backbone | Training data | Key notes | Score | Status |
|---|---|---|---|---|---|---|

Column definitions:
- **Date** — ISO format (YYYY-MM-DD)
- **Model / variation** — name or slug of the model version (e.g. `multilabel_234_v3/2`)
- **Backbone** — architecture name (e.g. `convnext_small`, `eca_nfnet_l0`)
- **Training data** — brief description (e.g. "clean train_audio", "clean + R1 pseudo-labels (238K clips)")
- **Key notes** — notable hyperparameters or conditions (e.g. "fine_tune(2) + fit_one_cycle(5)", "fp16", "hop=512")
- **Score** — the reported evaluation metric value (e.g. `0.805`). Leave blank if not yet scored.
- **Status** — one of: `kept`, `reverted`, `timed out`, `pending`

### Section 2: Decisions and reversals

A list of dated bullet points, one per decision or reversal. Format:

```
- YYYY-MM-DD: <what was decided or reversed>. Reason: <why>.
```

Examples:
- `- 2026-05-15: Committed to ConvNeXt-small as primary backbone. Reason: 50M params is the capacity sweet spot for ~85K training images.`
- `- 2026-05-22: Reverted SpecAugment + Mixup. Reason: scored 0.763 vs 0.765 baseline — marginal regression.`

### Section 3: Lessons

A living list of facts and corrections. Unlike the other two sections, this one is actively
updated rather than only appended to — if a lesson is superseded or proven wrong, edit it
in place rather than adding a contradicting entry.

Two sub-sections:

**Domain facts** — confirmed facts about the competition, data, tools, or ML techniques.
Format: `- <fact>. (Confirmed: YYYY-MM-DD)`

**Process corrections** — rules for how the AI agent should behave, derived from mistakes made.
Format: `- <rule>. Why: <the mistake that prompted this>.`

---

## How to make changes

### Adding an experiment

When a new model is trained and a score is reported, add a new row to the experiments table.
Do this immediately when the score is stated — do not wait until the end of a session.

### Adding a decision or reversal

When a significant choice is made (committing to an approach, reverting a change, abandoning
a direction), add a dated bullet to the Decisions section. "Significant" means: it would be
costly to undo, or a future session might otherwise try the same thing again.

### Updating lessons

When a new fact is confirmed or a correction is made:
1. Check whether an existing lesson already covers it. If so, update that entry rather than
   adding a duplicate.
2. If it is genuinely new, append it to the appropriate sub-section.
3. If an old lesson is contradicted by new evidence, edit it in place and add
   `(updated: YYYY-MM-DD)`.

---

## Update mechanism — AGENTS.md instruction

Add the following block to the repository's `AGENTS.md` file:

```
## Research log

1. At the start of every session, read `research/log.md` in full before taking any action.
2. After every scored result or committed decision is reported, update `research/log.md`
   before doing anything else.
```

Rule 1 ensures the agent loads full history at the start of each session without being asked.
Rule 2 makes updates habitual — the log is written at the moment of the event, not
reconstructed later from memory.

---

## Initial file template

When creating `research/log.md` for a new competition, start with this exact template:

```markdown
# Research Log

## Experiments

| Date | Model / variation | Backbone | Training data | Key notes | Score | Status |
|---|---|---|---|---|---|---|

## Decisions and reversals

_(none yet)_

## Lessons

### Domain facts

_(none yet)_

### Process corrections

_(none yet)_
```
