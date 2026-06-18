# Contributing

Thanks for your interest in `volforecast`. This project uses
[uv](https://docs.astral.sh/uv/) for environment and dependency management.

## Dev setup

```bash
# 1. Install uv (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the env and install the lean serve extras + viz + dev tooling.
#    (The research/LSTM/TensorFlow extra is intentionally NOT installed.)
uv venv
uv pip install -e ".[data,viz,dev]"
```

`uv venv` creates `.venv/`; prefix commands with `uv run` to use that env
without activating it, or activate it directly.

## Quality gates

These are exactly what CI runs (see `.github/workflows/ci.yml`). Run them locally
before opening a pull request:

```bash
uv run ruff check src                                              # lint
uv run mypy src                                                    # types (strict)
uv run pytest -q -m "not research and not slow" \
  --cov=volforecast --cov-report=term --cov-fail-under=85          # tests + coverage
```

- **Lint** (`ruff`) must pass.
- **Types** (`mypy --strict`) must pass.
- **Tests** (`pytest`) must pass with **coverage ≥ 85%** (the gate also lives in
  `[tool.coverage.report] fail_under` in `pyproject.toml`).

CI runs the full matrix on Python 3.11, 3.12, and 3.13. The research-only LSTM
arm (TensorFlow) is never installed or imported on the serve path or in CI.

## Commit hygiene

- Use clear, present-tense commit messages.
- Keep trailers clean: do not add co-author or generated-with attribution
  trailers to commits or pull requests.

## Pull requests

- Branch off `main`; keep PRs focused.
- Make sure the three quality gates above are green locally.
- Update `CHANGELOG.md` (under `[Unreleased]`) when behaviour changes.
