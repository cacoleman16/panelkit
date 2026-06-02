# Publishing panelkit to PyPI

## Readiness check (verified locally)

| check | status | notes |
|---|---|---|
| Name `panelkit` available on PyPI | ✅ | `GET pypi.org/pypi/panelkit/json` → 404 |
| sdist builds | ✅ | `maturin build --sdist` |
| wheel builds (abi3, py39+) | ✅ | one wheel per platform covers Python 3.9+ |
| sdist is self-contained | ✅ | bundles all crates/*.rs, Cargo manifests, README, LICENSEs, GUIDE |
| `twine check` (sdist + wheel) | ✅ | both PASS |
| clean-room install + import + fit | ✅ | fresh venv, numpy-only runtime dep |
| metadata: name/version/requires-python | ✅ | `>=3.9`, `numpy>=1.21` |
| metadata: license | ✅ | `MIT OR Apache-2.0` + LICENSE-MIT / LICENSE-APACHE |
| metadata: project URLs | ✅ | Homepage / Repository / Documentation / Issues |
| long_description (README) renders | ✅ | `twine check` validates the Markdown |
| determinism / tests green | ✅ | 39 Rust + 25 pytest; clippy + fmt clean |

## What's left to actually publish (manual / one-time)

Version is set to **`0.1.0`** (`Cargo.toml [workspace.package]` + `pyproject.toml`).

1. Repo + metadata URLs are set to `github.com/cacoleman16/panelkit`.
2. **(If re-releasing) bump the version** — PyPI versions are immutable; you
   cannot overwrite `0.1.0` once uploaded.
3. **Set up Trusted Publishing on PyPI** (recommended, no tokens):
   - Create the project on PyPI (or use "pending publisher" before first upload).
   - Add a trusted publisher: this GitHub repo, workflow `release.yml`,
     environment `pypi`.
4. **Tag and push:** `git tag v0.1.0 && git push --tags`. The `release.yml`
   workflow builds wheels (linux/macos/windows, manylinux) + sdist and publishes
   via OIDC.
5. **Smoke-test from TestPyPI first** (optional but advised):
   `maturin publish --repository testpypi ...`, then
   `pip install -i https://test.pypi.org/simple panelkit`.

## Manual publish (alternative to CI)

```bash
# Build everything, then upload with an API token.
maturin build --release --manifest-path crates/pypanelkit/Cargo.toml --sdist --out dist
twine check dist/*
twine upload dist/*        # prompts for token, or use ~/.pypirc / TWINE_* env
```

## Notes / caveats

- **Linux wheels** must be built in a manylinux container (the `release.yml`
  job uses `maturin-action` with `manylinux: auto`); a bare `maturin build` on a
  dev box produces a non-portable linux tag.
- **abi3**: we build against the stable ABI (`abi3-py39`), so a single wheel per
  platform serves all Python ≥ 3.9 — no per-minor-version matrix needed.
- The `panelkit/panelkit` GitHub URLs in metadata are placeholders; set them to
  the real repository before release.
