# Slate diversity local tests

This crate compiles the dependency-free selection engine used by Home Mixer's
`SlateDiversitySelector`. It does not compile the Home Mixer adapter or its
internal X dependencies.

Run from the repository root:

```sh
CARGO_TARGET_DIR=/tmp/xai-slate-diversity-target \
  cargo test --manifest-path local-tests/slate-diversity/Cargo.toml
```

The external target directory is required because this checkout's directory
name contains `:`, which macOS cannot represent as one entry in
`DYLD_FALLBACK_LIBRARY_PATH`.
