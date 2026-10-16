# Shared code index

Refresh this checkout with the codebase-memory `index_repository` tool, using
the project name `mlir-cim22-linux-usb-runtime`, full mode and persistence.
`.cbmignore` defines repository-relative scope; local build outputs and wheels
are already excluded by Git ignore rules and the indexer's skip rules.

The compressed SQLite graph stores relative source paths. Before committing an
export, set its project root to `.` and remove machine-specific checkout and
Git-directory metadata from Branch nodes and edges. Compact the database and
recompress it so removed values do not remain in unused SQLite pages; update
the artifact size metadata. Keep the live local index bound to the actual
checkout. Exported metadata must not contain developer home or worktree paths.

The artifact records the commit used for indexing; committing the artifact
itself necessarily creates a later commit. Resolve or refresh the local index
before relying on it. Use coverage checks and read reported partial ranges;
an index with no recorded gap is still best-effort evidence.
