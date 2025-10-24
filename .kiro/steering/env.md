---
inclusion: always
---
# .cursor/rules/env.mdc
---
description: "Entornos simplificados con micromamba"
alwaysApply: true
---
## Environment

### Rules
- One micromamba environment per project.
- Python dependencies declared in **pyproject.toml**.
- Basic development tools (mkdocs) declared in the `dev` group of pyproject.toml.
- Fix critical versions; otherwise use compatible specifications (`numpy >=1.26,<2.0`).
- Document randomness seeds (`--seed`, environment variable, or configuration file) for numpy, torch, random.
- When adding or updating dependencies, edit `environment.yml` first.
- Document new dependencies and their justification in commit messages.
- **When installing a new library:**
    - During development, you can install new dependencies with `pip install` for speed, but you must add them to `environment.yml` immediately.
- All changes to dependencies must be documented in `CHANGELOG.md` and in commit messages.

### Checklist
- [ ] `micromamba create -n env -f environment.yml` reproduces the build.
- [ ] Each model/persistent file embeds the commit hash in the metadata.
- [ ] `pip install -e .[dev]` installs the basic development tools.

## Dependency Management
- Use micromamba for environment management
- Keep environment.yml up to date with all dependencies
- Use conda-forge as the default channel
- Install new dependencies using `micromamba install` during development
- Document all dependency changes in CHANGELOG.md

## Environment Configuration
- Use Python 3.11
- Keep environment.yml up to date
