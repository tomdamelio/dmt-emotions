---
inclusion: always
---
# .cursor/rules/docs.mdc
---
description: "Documentation – MkDocs-Material & thorough docstrings"
alwaysApply: true
---
## Documentation

### Rules
- Auto-build docs with **MkDocs-Material** (`mkdocs.yml` in root).
  Deploy via GitHub Pages on each push to `main`.
- Docstrings: Google style with type hints; rendered into API pages by mkdocstrings.
- README.md must include **TL;DR**, install snippet, and 30-second usage demo.
- Keep tutorials / notebooks in `docs/examples/` and test them with **pytest-nb**.
- Periodically update `CHANGELOG.md` to reflect all notable changes, following [Keep a Changelog](mdc:https:/keepachangelog.com/en/1.0.0) and SemVer best practices.

### Checklist
- [ ] `mkdocs build --strict` succeeds locally.
- [ ] New public symbols include docstrings prior to merge.
