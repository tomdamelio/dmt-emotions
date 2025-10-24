---
inclusion: always
---
# .cursor/rules/git.mdc
---
description: "Version control – Conventional Commits, trunk‑based flow & DVC (Google Drive remote)"
alwaysApply: true
-----------------
## Git Practices

### Rules

* Use **Conventional Commits** (`feat: …`, `fix: …`, `docs: …`). Enforced by *commitlint* pre‑commit hook:

```text
# Examples of Conventional Commits
feat(preprocessing): add band-pass filter for EEG
fix(physio): correct EDA sampling rate bug
docs(readme): add mermaid pipeline diagram
chore(ci): pin mne==1.7 in GitHub Actions

```

* When you finish a consistent work block, invoke `Cursor: AI Commit` to suggest whether it is a good time to commit and generate a message according to the Conventional Commits standard.
* Short‑lived feature branches; rebase onto `main`; fast‑forward merges only.
* `main` stays green (tests + lint + type‑check pass in CI).
* Tag releases with **SemVer** (`vX.Y.Z`) and generate changelog via **release‑please**.
* Large data files should be stored externally (Google Drive, cloud storage) and referenced in documentation.
* Keep repository lightweight by avoiding large binary files in Git history.

### Checklist

* [ ] Today's commits follow the Conventional Commits spec.
* [ ] `git describe --tags --dirty --always` yields a tag or commit for every build.
* [ ] Repository size remains manageable without large binary files.
