---
inclusion: always
---
# .kiro/steering/git.md
---
description: "Git – DataLad, Commits Atómicos y 'El porqué' de los cambios"
alwaysApply: true
---
## Git & Provenance Practices

### Rules
- **DataLad Protocol**: **Never** commit large binary files (`.nii`, `.h5`, `.pth` > 10MB) to standard Git.
- **Atomic Scientism**: Do not mix scientific changes (logic/params) with cosmetic changes (formatting/typos) in the same commit.
- **The "Why" Mandate**: For commits changing scientific parameters (e.g., "Change smoothing kernel to 6mm"), the commit message body must explain the **scientific rationale**, not just the action.
- **Format**: Use Conventional Commits (`feat:`, `fix:`, `docs:`, `exp:` for experiments).

### Checklist
- [ ] No binary files in `git status`.
- [ ] Commit messages explain *why* a change was made, especially for parameters.
- [ ] `git describe --tags` allows tracing results to a specific code version.