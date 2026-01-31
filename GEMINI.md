# Project Context & Guidelines

## 1. Development Standards
- **README.md Synchronization:**
  - After every code change, ensure `README.md` is updated to reflect the current state and functionality of the project.
- **Tooling:**
  - Run `ruff check .` and `ruff format .` frequently.

- **Python Version:** Target **Python 3.14**.
- **Modern Type Hints:**
  - Use `X | None` instead of `Optional[X]`.
  - Use built-in generics (`list`, `dict`, `tuple`, `set`) instead of `typing.List`, `typing.Dict`.
  - Use `type` alias keyword for type definitions.

## 2. Code Philosophy
- **Simplicity & Readability:**
  - Write short, expressive, and easy-to-read code.
  - **Avoid over-engineering and excessive abstraction.** Focus on clear, direct solutions.
- **No Dead Code:**
  - Remove unused functions, variables, and commented-out code immediately. Keep the codebase clean.
- **Comments:**
  - **Strictly limit comments.** Only comment on *why* a complex decision was made. Avoid clutter.

## 3. Architecture & Best Practices
- **Functional Style:**
  - Prefer pure functions for algorithmic logic (e.g., crossovers, mutations, selections).
  - Use immutable data structures (e.g., `dataclasses(frozen=True)`) where possible to reduce side effects and improve predictability.
- **Minimal Exceptions:**
  - Do not overuse `try/except` blocks.
  - Let standard Python exceptions surface normally unless specific handling is critical for the application's robustness or user experience.

## 4. Package Management
- **Tooling:** This project uses **uv**.
- **Adding Dependencies:** Use `uv add <package>` instead of `pip install`.
- **Syncing:** Always ensure `uv.lock` is respected.

## 5. Experimentation & Testing
- **Testing Code:** Code changes should be validated by running experiments via `pipeline.sh`. This helps ensure that algorithmic changes and modifications do not negatively impact the overall system behavior and performance.