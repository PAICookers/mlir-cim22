# Project Engineering Rules

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
