# Shared Codex Rules

This is the shared rule source for the main agent and all role agents. Read it
before planning or editing. Role TOML files add responsibilities; they do not
override these rules.

## Engineering

- Understand the real flow and existing mechanisms before editing. Prefer
  upstream LLVM/MLIR facilities and the smallest working change.
- Keep semantic values, layout, mapping, packets, artifacts, transport, and
  hardware execution as separate contracts.
- Use Python for capture, runtime orchestration, independent oracles, and
  fixtures. Keep production lowering, target semantics, and frame codecs in
  MLIR/C++; do not duplicate them in Python.
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

## Python

These rules apply to project-owned Python source, tests, and examples. Keep
supplier, generated, and third-party code under their existing handling rules.

### Version And Formatting

- The Python runtime supports only Python 3.12. Declare
  `requires-python = ">=3.12,<3.13"` in runtime package metadata and run runtime
  imports and tests with that interpreter, including after annotation changes.
  Do not add older-version compatibility shims or use Python 3.13+ APIs.
- Use four spaces, 80-column formatting, and Ruff. Preserve the existing Ruff
  lint rules; use target `py312` when checking runtime source and its tests or
  examples.
- Use descriptive `snake_case` for modules, functions, parameters, and variables;
  `PascalCase` for classes; `UPPER_SNAKE_CASE` for constants; and a leading `_`
  for private names. Group imports as standard library, third-party, then
  project modules, following Ruff's import ordering.

### Types And Documentation

- Annotate public APIs and nontrivial internal functions, including return
  types. Prefer `list[T]`, `dict[K, V]`, and `T | None`; import abstract container
  types such as `Sequence` and `Mapping` from `collections.abc`. Use precise
  types instead of unexplained `Any` or blanket type-checker suppressions.
- Do not use `from __future__ import annotations`. Quote forward references
  and annotation names imported only under `TYPE_CHECKING`, unless those names
  are also available at runtime. Check annotations import successfully on the
  applicable supported interpreter.
- When runtime code needs type aliases or generics, prefer Python 3.12's
  `type Alias = ...` and type-parameter syntax.
- Public NumPy arrays use explicit dtypes and `numpy.typing.NDArray`. Document
  tensor dtype, shape, layout, ownership, and lifetime where they are part of an
  API contract; validate them at runtime before crossing native/ABI boundaries.
  Do not silently cast or reshape invalid inputs to make a call succeed.
- Use concise Google-style docstrings for public modules, classes, and APIs.
  Add `Args`, `Returns`, and `Raises` sections when needed to explain the
  contract. Comments explain constraints and reasons rather than restating code.

### Errors, Resources, And Side Effects

- Validate external inputs with explicit exceptions, not `assert`. Catch
  specific exceptions, preserve useful context with `raise ... from ...`, and
  do not hide failures with bare `except` or an unexplained `except Exception`.
- Avoid mutable default arguments and shared mutable class attributes. Use
  context managers or `try/finally` to release files and native runtime handles;
  keep resource ownership and cleanup explicit.
- Library code must not print or configure global logging. Do not run
  compilation or inference at import time. Use module loggers with lazy logging
  arguments; put script execution behind a `main()` and an
  `if __name__ == "__main__":` guard.
- Reuse existing project helpers and dependencies. Keep Python capture and
  runtime orchestration within the compiler/runtime boundary in Engineering;
  do not invent a second lowering path or frame codec.

### Verification

- For runtime changes, select Python 3.12 explicitly in the runtime environment.
  Run focused tests and import smoke checks first; expand to the relevant runtime
  suite when shared behavior changes. Keep compiler/software checks distinct
  from board evidence.
- Run `ruff check` and `ruff format --check` on touched project-owned Python
  files. Select `--target-version py312` for runtime checks. Do not reformat
  unrelated or supplier code, or add a second lint policy that conflicts with
  existing rules.
- For rules/docs-only changes, check `git diff --check`, links, and consistency
  with Python metadata and Ruff configuration. Do not claim source execution
  or runtime validation when only documentation was checked.

## Task Records

- For nontrivial work, maintain a local `tasks/todo.md`, creating it when absent.
  Record scope, workspace choice, ownership, dependencies, and verification
  before implementation; update status and review after each phase.
- Read `tasks/lessons.md` when present. After a user correction, record one
  reusable prevention rule, creating the local file when needed.
- Keep these records compact and local. They support the work but do not replace
  shared rules or reviewed project documentation, and are not automatically
  included in commits.

## Knowledge Sharing And Privacy

These rules govern project notes, supplier/hardware evidence, personal working
notes, and any archive or manifest derived from them. They do not grant rights
to redistribute supplier or third-party material. `AGENTS.md` points agents to
this file; do not maintain a second, conflicting copy of this policy.

### Classification

Every new or changed knowledge artifact must be classified by its owner before
it is staged, copied to a shared store, or sent to another person. Classification
describes the strictest applicable handling requirement; it does not establish
ownership or redistribution permission.

- `unclassified`: default for new or inherited material. Treat as local-only;
  do not stage, publish, upload, or include in an archive until reviewed.
- `personal`: an individual's scratch notes, drafts, credentials, private
  commentary, or other material not intended for the project team. Keep it
  outside the shared repository and shared stores.
- `restricted`: supplier-confidential, third-party-controlled, regulated, or
  otherwise access-limited material. Share only through a separately approved
  access-controlled store after confirming the owner permits that use.
- `team`: project material approved for all members who can access the team's
  private repository. This is not private from other authorized repository
  members.
- `public`: material explicitly approved for unrestricted publication. A
  public repository, release, or external post requires the owner's explicit
  publication approval.

When an artifact combines classes, use the most restrictive class for the
whole artifact. An index, filename, summary, screenshot, hash manifest, or
archive can reveal sensitive information too; classify metadata and derivatives
before sharing them. A hash proves byte identity only, not safety, provenance,
or permission to redistribute.

### Storage And Ownership

- Keep personal material outside the repository. Do not create a per-member
  folder inside a shared checkout as a privacy mechanism.
- Git ignore rules only reduce accidental staging. They do not protect data
  from people who can read the checkout, its backups, or repository history.
- Git access control applies to the repository, not individual files. Before
  using a remote for `team` material, verify that it is private, identify its
  access owner, and confirm that its membership matches the intended audience.
  If a member must be excluded from some files, use a separate repository or an
  approved store with narrower ACLs.
- `docs/project-notes/**` is the canonical location for reviewed, reusable
  project knowledge and decisions, not for private scratch work or raw supplier
  copies. Record claims with provenance and evidence state under the existing
  documentation rules.
- Treat supplier and third-party material as read-only. Do not copy, transform,
  archive, or redistribute it until the source owner, applicable terms, and
  intended audience permit that action. A local copy or an untracked Git status
  is not evidence of permission.
- Put approved large or restricted evidence in the team's separately
  access-controlled artifact store. Keep only a reviewed manifest in Git, and
  only when its paths, names, hashes, producer details, and retrieval
  instructions are safe for the Git audience. Do not put access tokens or
  credentials in the manifest.

### Promotion And Sharing Workflow

Before moving local knowledge into a shared location:

1. Identify its creator/owner, source, intended audience, classification, and
   permission to redistribute. If any of these are unclear, leave it local and
   ask the owner; do not infer permission from possession or project relevance.
2. Share only the minimum useful content. Prefer a concise, sourced project note
   over copying a supplier document, raw personal notes, or an entire directory.
   Preserve technical evidence status, applicability, and unresolved conflicts.
3. Have the author review the proposed shared text and a project maintainer
   review its classification, provenance, and destination. Obtain explicit
   publication approval for `public`; obtain confirmation from the access
   owner before placing `team` or `restricted` material in a remote store.
4. Inspect the exact outgoing file list, symlink targets, archive contents,
   metadata, and generated derivatives. Check for credentials, personal
   identifiers, private paths, unpublished results, and unrelated local files.
   Use the approved secret-scanning tool when available, but do not treat a
   clean scan as proof of redistribution rights or absence of all sensitive
   content.
5. Stage explicit reviewed paths only. Never use broad staging such as
   `git add docs/` or `git add -A` to share knowledge. Review both the staged
   diff and staged file list before committing or pushing.
6. Push only to the verified destination and intended audience. Recheck remote
   visibility and membership when the repository or artifact-store ACL changes.

Do not modify existing local-only or untracked material as part of a sharing
plan until each item has been classified and its owner has authorized the
specific migration. A plan or request to improve sharing does not authorize
staging, committing, uploading, publishing, or changing ACLs.

### Exposure And Incident Handling

If credentials or material appear to have been shared with an unintended
audience, stop further distribution and notify the repository/access owner and
the data owner promptly. Revoke or rotate exposed credentials first. Preserve
enough incident details to identify affected paths, commits, archives, stores,
and audiences without reposting the sensitive content. Deleting a file does
not remove it from Git history, clones, backups, or downloaded archives; history
rewrites and remote cleanup must be coordinated with the repository owner and
affected collaborators. Do not independently rewrite shared history.

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

Stop only the affected scope when a missing or disputed fact could change a
public ABI, destructive hardware action, deployable artifact, or shared
contract. Escalate with the exact missing evidence and a safe probe. Continue
independent software-only work when the uncertainty is isolated and labeled.
