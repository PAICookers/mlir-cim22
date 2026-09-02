# Shared Codex Rules

This is the shared rule source for the main agent and all role agents. Read it
before planning or editing. Role TOML files add responsibilities; they do not
override these rules.

## Engineering

- Understand the real flow and existing mechanisms before editing. Prefer
  upstream LLVM/MLIR facilities and the smallest working change.
- Keep semantic values, layout, mapping, packets, artifacts, transport, and
  hardware execution as separate contracts.
- Treat Python as an independent oracle or fixture source, not a second
  production compiler or frame codec.
- Treat hardware claims conservatively. Fixtures, host relays, and software-only
  agreement do not prove board execution or deployability.
- Record provisional assumptions with their CTQ/HWQ, replacement boundary, and
  non-deployable status. Never silently fill an unresolved contract.
- Preserve user changes and supplier evidence. Avoid destructive Git operations.
- In change, build, fix, or test tasks, perform in-scope local edits and
  non-destructive validation without pausing for routine permission approval.
  Inherit the parent session's access policy instead of narrowing it in a role
  TOML. Stop only when an action needs authority beyond the user's request or is
  destructive, external, costly, or materially expands scope.

## Collaboration

- Consider a specialist when work has an independently bounded scope, benefits
  from a second technical view, or requires independent verification. Keep a
  small, tightly coupled change with one owner.
- The architect owns shared contracts, milestone gates, cross-module decisions,
  and final integration. Specialists own bounded implementation or evidence
  packages and do not silently expand their scope.
- Parallel work requires disjoint `allowed_files`, explicit `blocked_files`,
  dependencies, a frozen interface, and scope-local verification. One
  integration owner runs the broader regression.
- Cross-boundary ambiguity returns to the architect with the smallest useful
  evidence probe. Do not guess or redesign another agent's interface.
- Every handoff reports `status`, `summary`, `artifacts`, `changed_files`,
  `commands_and_results`, `blockers`, and `next_actions`. Verification also
  reports verdict, reproduction, evidence scope, and unverified boundaries.

## Stop And Escalate

Stop only the affected seam when a missing or disputed fact could change a
public ABI, destructive hardware action, deployable artifact, or shared
contract. Escalate with the exact missing evidence and a safe probe. Continue
independent software-only work when the uncertainty is isolated and labeled.
