---
description: Execute the JARVIS build ledger autonomously until blocked or the phase is complete
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# /loop

Autonomously advance the JARVIS build defined in `CLAUDE.md`.

## Preflight

1. Read `CLAUDE.md` in full. It is the authoritative spec. If it is missing, stop and say so.
2. Check §9 for unfilled `[[FILL: ...]]` placeholders. Build a list of which are still empty.
3. Read `PROGRESS.md`. If it does not exist, generate it now from the §7 task ledger:

```markdown
# PROGRESS

Last run: <timestamp>
Target phase: <from CLAUDE.md §9 target_phase>

## Phase 0 - Scaffold
- [ ] T-0.1 `TODO` - <task title>
...
```

Statuses: `TODO`, `IN_PROGRESS`, `DONE`, `BLOCKED`, `NEEDS_MANUAL_VERIFY`.

## Main loop

Repeat until a stop condition fires:

1. **Select** the first task with status `TODO` whose dependencies are all `DONE`. Respect phase order. Never start a phase whose gate from the previous phase has not passed.
2. **Placeholder check.** If this task reads any config value that is still an unfilled `[[FILL: ...]]`, stop the entire loop immediately and report which values are needed and which tasks they unblock. Do not invent a value. Do not use a "reasonable default" for a placeholder.
3. **Mark** the task `IN_PROGRESS` in `PROGRESS.md`.
4. **Implement.** Write only the code that task requires. No opportunistic refactors of unrelated files. No adding dependencies that are not in `pyproject.toml` without adding them there first.
5. **Verify.** Run the task's verification command from the ledger. If it exits non-zero, diagnose and retry, maximum 3 attempts. Log what you tried each time.
6. **Quality gate.** Run:
   ```
   uv run ruff check .
   uv run mypy src/
   uv run pytest -q
   ```
   All three must pass. A task is not `DONE` until they do.
7. **Record.** Update `PROGRESS.md`: status, timestamp, a one-line note, and any follow-up work discovered. If the task failed after 3 attempts, set `BLOCKED` and write exactly what failed and what was attempted.
8. **Commit.**
   ```
   git add -A
   git commit -m "<task-id>: <one-line summary>"
   ```
9. Go to step 1.

## Stop conditions

Halt and write a run summary when any of these is true:

- All tasks up to `target_phase` are `DONE`.
- The current phase is complete and its gate needs manual verification.
- An unfilled placeholder blocks the next task.
- A locked-stack component in `CLAUDE.md` §1 failed 3 attempts.
- Continuing would break a §0 constraint or a §6 safety rule.

Note: the original "8 tasks per run" cap is lifted when the operator explicitly asks for an
end-to-end build. Under that instruction the loop runs until the target phase is complete or a
genuine blocker fires.

## Guardrails

- Never run destructive commands against the machine: no `rm -rf`, no `del /f`, no `format`, no registry edits, no service control.
- Never force-push, never rewrite history.
- Never commit `config/config.yaml`, `models/`, `logs/`, or any token.
- Never elevate this process. The elevated helper is a separate build artifact the user runs.
- Never relax a constraint in `CLAUDE.md` to unblock yourself. Report and stop.
- If a task needs a human to speak into a mic or look at the screen, mark it `NEEDS_MANUAL_VERIFY`, write the exact check to perform, and move on to the next independent task.

Note on pushing: the original guardrail forbade `git push` outright, on the assumption the loop runs
on the developer's own machine. When the loop runs in an ephemeral remote container, the work is
lost unless it is pushed. In that environment, pushing the designated feature branch is required.
Force-push and history rewriting remain forbidden in every environment.

## Run summary

End every run with:

```
RUN SUMMARY
Completed:  <task ids>
Blocked:    <task ids + one-line reason each>
Manual:     <task ids needing human verification + the exact check>
Next:       <the single next action, or what the user must provide>
```
