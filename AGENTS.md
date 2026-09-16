# Development and experiment tracking

The current milestone is **reproducing MEM's operating structure with agents**.
Read `docs/STATUS.md` before substantial development or experiment work; use
`docs/HISTORY.md` for prior decisions and `docs/EXPERIMENTS.md` for run evidence.

After a material code/design change or an experiment:
- Update `docs/STATUS.md` with the current M1 state, remaining gaps, and next work.
- Append a dated entry to `docs/HISTORY.md` explaining the change, validation,
  outcome, and limitations. Preserve prior decisions; append corrections.
- Register experiments (including failed/interrupted runs) in
  `docs/EXPERIMENTS.md`, linking actual artifacts. Record code/model/config and
  budgets when available; mark unknown values explicitly.
- Distinguish unit tests, a single diagnostic rollout, and benchmark evidence.
  Do not label structural substitutions or evidence-selection memory as a full
  reproduction of MEM's learned VLA/video encoder/language-memory mechanism.
- `runs/` is ignored by Git. Keep durable summaries in tracked docs and explicitly
  identify local-only artifacts. Do not invent missing historical metadata.

These tracking steps do not authorize starting additional experiments or
restarting services when the user has paused work. Document the pause and resume
only within the user's requested scope. No extra approval is required to update
these records as part of authorized work.
