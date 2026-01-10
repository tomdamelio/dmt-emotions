---
inclusion: always
---
# .kiro/steering/style.md
---
description: "Style – Ruff, Black y Naming Neuro-Semántico"
alwaysApply: true
---
## Code Style

### Rules
- **Uncompromising Formatting**: Use `black` (or `ruff format`) for all Python code. No debates on style.
- **Neuro-Semantic Naming**: Use domain-specific variable names.
    - *Good*: `subject_id`, `bold_signal`, `repetition_time_tr`.
    - *Bad*: `id`, `data`, `x`, `t`.
- **Type Hints**: All function signatures must include type hints.
- **Clean Code**: Remove dead code immediately. Do not comment out old logic; delete it (Git remembers).

### Checklist
- [ ] Code is formatted with Black.
- [ ] No single-letter variables (except `i`, `j`, `k` in short loops).
- [ ] Imports are sorted and clean.
