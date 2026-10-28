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

- Follow the range declared in `python/pyproject.toml`; Python 3.12 is the
  validated baseline for runtime imports and tests. Other interpreter and
  dependency combinations are unverified until tested, so keep source within
  the declared range and record the tested combination.
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
- When runtime code needs type aliases or generics, prefer syntax supported by
  the declared Python range; version-specific syntax requires matching metadata
  and validation.
- Use `object` for values that require narrowing, including deliberately invalid
  test inputs. Use `Any` at dynamic third-party interfaces when their operations
  cannot be described accurately with the available types.
- Public NumPy arrays use explicit dtypes and `numpy.typing.NDArray`. Document
  tensor dtype, shape, layout, ownership, and lifetime where they are part of an
  API contract; validate them at runtime before crossing native/ABI boundaries.
  Do not silently cast or reshape invalid inputs to make a call succeed.
- Use concise Google-style docstrings for public modules, classes, and APIs.
  Add `Args`, `Returns`, and `Raises` sections when needed to explain the
  contract. Comments explain constraints and reasons rather than restating code.
- Keep `cim22.__init__` limited to user-facing APIs. Import internal tables,
  helpers and native-library discovery from their owning modules.

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

- For runtime changes, run focused tests and import smoke checks with the
  validated interpreter first; expand to the relevant runtime suite when shared
  behavior changes. Other interpreters remain unverified. Keep compiler/software
  checks distinct from board evidence.
- Run `ruff check` and `ruff format --check` on touched project-owned Python
  files. Select `--target-version py312` for runtime checks. Do not reformat
  unrelated or supplier code, or add a second lint policy that conflicts with
  existing rules.
- Prefer the host `ruff` executable for formatting and linting. Do not add Ruff
  to project Python dependencies unless the user explicitly requests it; if
  host Ruff is unavailable, report that formatting could not be run.
- For rules/docs-only changes, check `git diff --check`, links, and consistency
  with Python metadata and Ruff configuration. Do not claim source execution
  or runtime validation when only documentation was checked.

## Build And Tests

- Keep one Python project under `python/`; do not restore root Python metadata
  or a second setuptools build. scikit-build-core drives the root CMake project
  and packages `cim22`, the optimizer and compiler/runtime libraries together.
- Default to Release with tests and the ONNX diagnostic importer disabled.
  Debug is a standard CMake build type, not a separate runtime ABI or backend.
  Enable `MLIRCIM22_INCLUDE_TESTS` and `MLIRCIM22_ENABLE_ONNX` explicitly for
  native testing; use the Python `dev` group for pytest and fixture dependencies.
- Use the pinned Protobuf/protoc pair from FetchContent. ONNX is optional and
  shares these targets; do not reintroduce system-package selection branches.
  USB builds use Linux libusb or Windows x64 CyAPI; `CIM22_ENABLE_USB=OFF`
  builds software support only. USB artifacts must not fall back to software.
- Reinstall the editable package after native, TableGen, schema or CMake edits
  with `uv sync --project python --reinstall-package mlir-cim22`, using
  the same build directory and required CMake options. Build-environment changes
  may reset cached settings. `uv --offline` does not disable CMake downloads.
  Finish relinking before tests load libraries, then restart applications.
- Run pytest from `python/`; use native fixtures only where tests need them.
  Reuse immutable compiled fixtures for boundary checks. Keep independent
  numeric/layout oracles and mutation tests that cover distinct semantics.
- lit owns the native suite and invokes Python unittest helpers through `.test`
  wrappers. Use exit status for unittest success and FileCheck for compiler or
  diagnostic output. Resolve LLVM tools explicitly through lit substitutions.
  Keep native test assertions enabled in Release as well as Debug.
- Use host `ruff` and `clang-format`; exclude generated and supplier sources.
  After packaging changes, inspect sdist contents and run installed inference
  outside the source directory without native-library environment overrides.
- `torch.compile` is the application entry point. Keep readonly fixed weights
  constant for CIM lowering and dynamic weights on Host. Reuse AOTAutograd for
  mutations/aliases; do not recreate output schemas or standalone loading APIs
  for excluded application scenarios. Native Session returns flat tensors.
- Version serialized model data only with `ModelArtifact.schema_version`;
  do not duplicate fixed transaction versions in the schema. When deleting
  fields in any schema message, do not reserve removed fields and renumber the
  remaining fields compactly unless explicitly requested otherwise. Regenerate
  bindings, update readers/writers/tests and recompile models after such changes;
  compact renumbering does not preserve old artifact compatibility.
  During development, emit `schema_version=1` without enforcing it. Python
  Protobuf bindings live under `python/tests/_generated` and are test-only;
  regenerate them with the CMake target after schema edits.
- Keep `.cbmignore` repository-relative and prefer a narrow allowlist of source,
  build, test and schema directories; exclude local documents, models, reports,
  task records and metadata by default.
  Refresh the actual worktree index when requested; inspect persisted graph
  metadata for machine paths before committing it. Keep local wheels, task
  records, logs and build outputs untracked; never stage a whole dirty tree.

### Codebase Memory

- Codebase Memory walks the repository filesystem, so untracked source and
  documentation are indexed unless excluded by `.cbmignore` or an indexer
  skip-list. Git tracking status is not an indexing boundary.
- Resolve the current-root index with `index_status(project="@current")` when
  supported; otherwise match `list_projects` by checkout root. Refresh that
  project with `index_repository`, `full` mode and persistence; never repoint
  another worktree's registered project.
- The persisted graph stores relative paths. Before sharing an export, remove
  checkout and Git-directory metadata, compact/recompress the SQLite graph, and
  verify that no machine paths remain. Keep the live index bound to this
  checkout.
- Keep `.codebase-memory/artifact.json` local. Index only reviewed source/build
  trees; use the same narrow tool allowlist as the source distribution.
- Check index coverage and read reported partial ranges before relying on graph
  results. A clean coverage result is best-effort evidence, not proof of
  completeness.

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
