---
inclusion: always
---
# .cursor/rules/structure.mdc
---
description: "Standard folder layout – pyproject & src-layout"
alwaysApply: true
---
## Project Structure

```text
campeones_analysis/                 # ← project root (Git repo)
├── .github/workflows/              # CI pipelines (simplificadas)
│   └── ci.yml
├── data/                           # ★ BIDS dataset (stored externally, .gitignored)
│   ├── sub-01/                     # raw data in BIDS layout
│   ├── derivatives/                # ICA, filters, features (BIDS-Derivatives)
│   └── README                      # how to obtain full dataset (if not public)
├── configs/                        # YAML-or TOML-based experiment configs
│   ├── preprocessing_eeg.yaml
│   └── model_glhmm.yaml
├── docs/                           # ★ MkDocs-Material sources / API docs
│   └── index.md
├── notebooks/                      # exploratory notebooks 
│   ├── 00_overview.ipynb
│   ├── eeg/
│   └── ml/
├── src/                            # importable Python package (src-layout)
│   ├── campeones_analysis/         # ★ package name mirrors repo
│   │   ├── __init__.py
│   │   ├── eeg/
│   │   │   ├── preprocessing.py
│   │   │   └── features.py
│   │   ├── physio/
│   │   │   ├── preprocessing.py
│   │   │   └── features.py
│   │   │   └── read_xdf.py         # Conversión de XDF a BIDS
│   │   ├── behav/
│   │   │   └── parsing.py
│   │   ├── fusion/
│   │   │   └── multimodal_dataset.py
│   │   ├── models/
│   │   │   ├── train_ml.py
│   │   │   └── train_glhmm.py
│   │   └── utils/
│   │       └── io.py
├── scripts/                        # Scripts útiles
├── pyproject.toml                  # ★ PEP 621 metadata simplificada 
├── environment.yml                 # micromamba/conda env spec
└── README.md                       # project overview & 30-sec quick-start

```

### Rules
- PyProject-only: no setup.py; use PEP 621 metadata.
- README.md, .gitignore, pyproject.toml, LICENSE, CHANGELOG.md always present.
- Simplified structure to enable rapid iterations.

### Checklist
- [ ] src/project_name/__init__.py exists and exposes public API.
- [ ] Large data and results paths are in .gitignore (allowlist with ! as needed).
