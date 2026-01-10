---
inclusion: always
---
# .kiro/steering/modularity.md
---
description: "Modularity – Funciones puras y separación de I/O"
alwaysApply: true
---
## Modularity & Design

### Rules
- **Framing vs. Coding**: The user provides the architecture (Framing); the AI provides the implementation (Coding).
- **Pure Functions**: Prefer functions that take inputs and return outputs without side effects (no modifying global state).
    - *Bad*: `process_data()` (reads global variable).
    - *Good*: `processed = process_data(raw_array, config)`.
- **I/O Isolation**: Keep data loading/saving separate from computation logic. This enables easier testing.
- **Configuration**: No magic numbers inside code. Pass parameters via a Config object (Pydantic or frozen dataclass).

### Checklist
- [ ] No circular imports.
- [ ] Functions are short (< 50 lines) and single-purpose.
- [ ] "Magic numbers" are extracted to constants or config files.

