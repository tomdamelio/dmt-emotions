---
inclusion: always
---
# .cursor/rules/modularity.mdc
---
description: "Separation of concerns & modular design"
alwaysApply: true
---
## Modularity

### Rules
- Keep **I/O** and **pure computation** in separate layers.
- Functions ≤ 40 lines & single responsibility; split otherwise.
- No hidden globals: pass config explicitly or via frozen pydantic model.
- **DRY**: shared helpers live in `src/utils.py` or a dedicated sub-package.
- Package layout under `src/` follows **src-layout** with `__init__.py` exports.
- Use classes only for stateful abstractions; prefer dataclasses / attrs for simple containers.
- Automate quality checks, lint, type checking, and tests with Nox and pre-commit to ensure modularity and maintainability.
- All dev tools must be integrated into the project's reproducible workflow.

### Checklist
- [ ] Every module in `src/` can be imported without executing I/O.
- [ ] No circular imports (ruff rule `F401` disabled if unavoidable but documented).
- [ ] Nox and pre-commit run all defined quality checks and tests for the project.
