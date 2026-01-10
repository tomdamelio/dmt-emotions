---
inclusion: always
---
# .kiro/steering/env.md
---
description: "Entornos simplificados con micromamba"
alwaysApply: true
---
## Environment & Reproducibility

### Rules
- **Dependency Pinning**: Critical scientific libraries (numpy, pandas, mne, nilearn, torch) must be pinned to specific versions (e.g., `numpy==1.26.4`) in `environment.yml` or `pyproject.toml` to prevent "floating dependency" rot.
- **Seed Determinism**: All stochastic operations (ML models, simulations) must define a random seed at the entry point.
    - Explicitly log the seed used in the output metadata.
- **Continuous Audit**: If you import a package in a Python file, verify it exists in `environment.yml`. No "ghost imports".

### Checklist
- [ ] `micromamba create -f environment.yml` reproduces the build exactly.
- [ ] `pip install -e .[dev]` installs dev tools.
- [ ] Random seeds are configurable via config or args, not hardcoded inside functions.
