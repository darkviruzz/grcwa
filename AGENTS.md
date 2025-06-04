# Contributor Guide for Codex

## Overview
This repository contains the Python package **grcwa**. Source code lives in the
`grcwa/` directory, tests in `tests/`, and documentation in `docs/`. Common
tasks are defined in the `Makefile`.

## Development Environment
- Run `pip install -r requirements.txt` in a virtual environment.
- `make lint` runs `flake8 grcwa tests`.
- `make test` runs the pytest suite. Use `tox` to test against all supported
  Python versions.
- `make docs` builds the Sphinx documentation under `docs/`.
- Keep code style consistent with `.editorconfig` (4‑space indent, UTF‑8).

## Testing Instructions
- New code must include tests under `tests/`.
- Ensure `pytest` and `flake8` pass before committing.
- The CI configuration is located in `tox.ini` and the `Makefile`.

## Pull Request Guidelines
- Provide a clear description of the change.
- Include test results and mention any updated documentation.
- Ensure the PR works for Python 3.5–3.8 as covered by `tox`.

## Using Codex in this Repository
- Point Codex to specific files or functions when requesting changes.
- Include steps to run `make lint` and `make test` so Codex can verify its work.
- Break large tasks into smaller steps and consider open‑ended prompts for code
  cleanup or debugging help.

