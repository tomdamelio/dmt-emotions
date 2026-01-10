---
inclusion: always
---
# .kiro/steering/docs.md
---
description: "Documentation – MkDocs, Context Engineering y Diccionarios de Datos"
alwaysApply: true
---
## Documentation & Literate Programming

### Rules
- **Context Engineering**: Write the docstring **before** the code. The docstring must explain the scientific intent (the "Framing") so the AI can generate the implementation (the "Coding").
    - Style: Google Style Python Docstrings.
- **Data Dictionaries**: If code outputs tabular data (`.tsv`, `.csv`), you must generate a BIDS-compliant JSON Data Dictionary describing columns and units.
- **Changelog**: Maintain `CHANGELOG.md` following "Keep a Changelog".

### Checklist
- [ ] Every public function has a docstring explaining inputs, outputs, and *scientific reference* (if applicable).
- [ ] Tabular outputs have associated JSON descriptions.