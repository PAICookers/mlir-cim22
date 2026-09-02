# Project Engineering Rules

Before planning or editing, read `.codex/project-rules.md`. It is the shared
rule source for the main agent and the role agents under `.codex/agents/`;
role TOML files add responsibilities but do not override it. For independently
bounded work or independent verification, consider the matching specialist and
keep file ownership disjoint.

- Prefer the types, algorithms, analyses, `PatternRewriter`, TableGen, pass
  infrastructure, and test tools supplied by the currently locked LLVM/MLIR.
- Before writing a project helper, IR representation, or abstraction, search
  the existing project implementation and LLVM/MLIR 22.1.8 source.
- Add a project implementation only when upstream cannot express the project
  semantic. Record the gap in the plan or review, and keep the implementation
  local and minimal.
- Update `tasks/todo.md` with status and review immediately after every
  development phase.
- If implementation differs from the plan, revise the relevant spec,
  hardware/compiler note, and `tasks/lessons.md` in the same phase. Do not
  retain stale conclusions.
- Do not declare a phase complete until code, tests, and records agree.

### Code Documentation

- Add concise, language-appropriate comments or docstrings for key code
  sections, functions, objects, types, and modules when they explain the main
  function, non-obvious behavior, important constraints, or notable risks.
- Explain intent, assumptions, and reasons rather than restating code that is
  already clear. Keep documentation accurate when behavior changes.
- Use `TODO`, `FIXME`, `XXX`, and similar markers in the target language's and
  project's established comment style. Each marker must state the concrete
  follow-up, defect, or risk; never leave a bare label.
- When modifying an existing source file, improve missing or stale
  documentation in the touched area when it materially helps maintenance. Do
  not create unrelated comment-only churn.
