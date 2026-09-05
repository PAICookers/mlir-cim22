# RTL Frame Feedback

- [x] Create `fix/rtl-frame-feedback` from `main` in a separate worktree;
  preserve the uncommitted BF16 work.
- [x] Trace all three feedback defects to the RTL exporter and compare the
  supplier TB's weight bodies, off-chip route, and five-bit response index.
- [x] Add independent artifact regression checks and demonstrate baseline failure.
- [x] Fix weight bodies, per-core return configuration, and exact target sets.
- [x] Run regressions, regenerate the demo, and review code/test/docs agreement.
- [x] Commit, push, and open a PR against `main`:
  https://github.com/PAICookers/mlir-cim22/pull/1

Review: the independent checker rejects the original four-core artifact in
three separate runs: duplicate CIM address 168, non-off-chip return, and inactive
configuration recipients. LLVM/MLIR 22.1.8 Debug build and the complete
`check-mlir-cim22` suite pass (63/63), including 2/4/6/20-core exports. The default
demo's source and expected files remain byte-identical to the original.
clang-format 22.1.8 checks pass on the changed C++ files. Frozen DUT RTL and raw
replay logs are not in this checkout; software tests cannot establish RTL PASS
or board deployability.
Existing Git credential authentication is available for PR creation. CI work
is isolated on `ci/cpp-checks` and will not be included in this PR.
The fix is submitted from the personal fork because the current identity has
read-only access to the upstream repository. The BF16 worktree is unchanged.
