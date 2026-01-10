---
inclusion: always
---
# .kiro/steering/structure.md
---
description: "Structure – Layout BIDS, src-layout y separación de datos/código"
alwaysApply: true
---
## Project Structure & Ontology

<repo>/
├── src/<package>/                 # paquete importable
├── tests/                         # tests automatizados
├── configs/                       # configs (YAML/TOML/JSON) para experimentos/pipelines
├── scripts/                       # entrypoints y utilidades (sin lógica central)
├── docs/                          # documentación y ejemplos
├── data/                          # datos (no versionados), con README de obtención
├── outputs/ or results/           # derivados (gitignored), generados por pipeline
├── pyproject.toml                 # config de tooling/paquete
├── environment.yml                # entorno reproducible (si aplica)
├── README.md
├── CHANGELOG.md
├── LICENSE
└── CITATION.cff


### Rules
- **Immutable Input Zone**: The `data/raw` (or `data/bids`) directory is **Read-Only**. Never generate code that opens these files with write permissions (`w`, `a`, `r+`).
- **BIDS Isomorphism**: All neuroimaging data must follow BIDS.
    - Derived data goes to `data/derivatives/<pipeline_name>`.
    - Filenames must follow key-value pairs (e.g., `sub-01_task-rest...`).
- **Source Layout**: All scientific logic lives in `src/<package_name>/`.
    - Use `__init__.py` to expose a clean public API.
- **Cookiecutter Strategy**: When creating a new module, always create the triad simultaneously:
    1.  Implementation: `src/module.py`
    2.  Test: `tests/test_module.py`
    3.  Docs: `docs/api/module.md` (or docstrings)

### Checklist
- [ ] `src/<package_name>/__init__.py` exists.
- [ ] Large data paths (`data/`) are in `.gitignore` (except for small test samples).
- [ ] No hardcoded absolute paths; use relative paths or config files.