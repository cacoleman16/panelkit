# Security

## Reporting

Found a vulnerability? Please open a private security advisory on the GitHub repo
(Security → Report a vulnerability) rather than a public issue.

## Threat model

panelkit is a numerical library. It does **not** parse untrusted file formats,
make network calls, execute user-supplied code, or handle credentials/secrets at
runtime. The realistic surface is: malformed array input from the calling Python
program, the Rust↔Python FFI boundary, and the dependency tree.

## Review summary (v0.1.0)

| area | status |
|---|---|
| `unsafe` in panelkit source | **none** — `grep -rn unsafe crates/*/src` is empty; all `unsafe` lives in the vetted `pyo3`/`numpy` crates |
| Dependency advisories (`cargo audit`) | **clean** — pinned `pyo3`/`numpy` ≥ 0.25 (fixes RUSTSEC-2025-0020 in pyo3 < 0.24.1) |
| FFI boundary (`crates/pypanelkit/src/convert.rs`) | numpy arrays are **copied** into owned column-major `Mat` at the boundary (bounds-checked indexing); no borrowed buffers held across the rayon region |
| Input validation | treated indices, `treat_time`, panel shape, and NaN/inf are validated in the Python layer (raises `ValueError`), so bad input can't reach a Rust panic; the core also asserts as defense-in-depth |
| Integer/array sizing | sizes come from the caller's own arrays; no untrusted length fields. Extremely large inputs can OOM (a normal resource limit, not a vuln) |
| Determinism / RNG | the PRNG (`Xoshiro256pp`) is for bootstrap resampling **only** — it is *not* cryptographic and must not be used for secrets |
| CI/CD workflows | no untrusted `${{ github.event.* }}` interpolated into `run:` steps (no script injection); release uses PyPI **Trusted Publishing (OIDC)** — no long-lived tokens stored; the only elevated permission is `id-token: write` on the publish job |
| Supply chain | runtime Python dep is `numpy` only; wheels built in CI from tagged source and published via OIDC |

## Running the checks yourself

```bash
cargo audit                                   # dependency advisories
grep -rn "unsafe" crates/*/src                # should print nothing
cargo clippy --workspace --all-targets -- -D warnings
```

## Notes for users

- panelkit assumes the input panel is **complete and finite**; NaN/inf are
  rejected up front.
- The bootstrap RNG is seedable for reproducibility, not security.
