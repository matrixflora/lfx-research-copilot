# Contributing to LFX Research Copilot

## Development Setup

1. Fork the repository and clone your fork.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Verify the pipeline runs:
   ```bash
   python run_validation.py
   ```

## Coding Standards

- **Python version**: 3.10+
- **Style**: Follow PEP 8. Use 4-space indentation (the repository uses 2-space in some files — prefer 4 for new code).
- **Imports**: Group in order — standard library, third-party, local. One import per line.
- **Type hints**: Use Python type annotations for all function signatures.
- **Docstrings**: Every module and public function must have a docstring describing purpose, arguments, and return values.
- **Logging**: Use the `logging` module with module-level loggers. Do not use `print()` for operational output.

## Branch Naming

- `feature/<short-description>` — new features
- `fix/<short-description>` — bug fixes
- `docs/<short-description>` — documentation changes
- `refactor/<short-description>` — code restructuring

Keep branch names lowercase with hyphens as separators.

## Pull Request Workflow

1. Create a branch from `main` for your work.
2. Make changes and commit with clear, concise commit messages.
3. Ensure all existing tests pass:
   ```bash
   python -m unittest discover tests/ -v
   ```
4. Run the pipeline in quick mode to verify nothing is broken:
   ```bash
   python run_validation.py
   ```
5. Push your branch and open a pull request against `main`.
6. In the PR description, explain what the change does and why it is needed.
7. A maintainer will review your PR. Address any feedback before merging.

## Documentation Standards

- Public API functions must include docstrings (Google or NumPy style).
- Module-level docstrings should explain the module's role in the pipeline.
- When adding a new module, update the pipeline stage list in `src/pipeline.py`.
- For user-facing features, update `README.md` or the relevant `docs/` file.

## Bug Reporting

Open a GitHub issue with the following:

- **Summary**: Concise description of the bug.
- **Environment**: Python version, OS, RAM.
- **Steps to reproduce**: Exact commands run and input data used.
- **Expected behavior**: What should happen.
- **Actual behavior**: What actually happens, including full error output.
- **Relevant files**: Any module names, output files, or logs that help diagnose.
