# Contributing

Thank you for considering contributing to MARCD!

## Setup
- Create a virtual environment and install dev deps:
  - `pip install -e .[dev,full]`
- Install pre-commit hooks:
  - `pre-commit install`

## Style
- Run `ruff`, `black`, and `isort` before committing.
- Prefer small, focused pull requests with tests where possible.

## Tests
- Run `pytest -q`.
- For heavy components (e.g., diffusion), provide unit-testable adapters or smoke tests.

## Pull Requests
- Describe the motivation, approach, and any trade-offs.
- Link issues when applicable.
