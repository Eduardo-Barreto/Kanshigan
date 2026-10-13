# Phase 7: PR

Open the PR once the gate and `review-judgment` are both clean. Use `/open-pr` or `gh pr create` directly; a `PreToolUse` hook re-runs `bun run doctor` and re-checks the reviewer's verdict against the current scope hash before letting it through.

- Title: under 70 characters, Conventional Commits format, in English, matching this repo's commit convention.
- Description: in Portuguese (pt-BR), following section 11 of `docs/anti-slop-guidelines.md` (PRs, Commits, and Review Comments). Lead with what changed and why, for a reviewer who already knows the codebase.
- Trade-offs: always call out, in plain language, anything cut, deferred, or accepted as a known limitation, even a small one. Anything `review-judgment` filed as low or non-blocking belongs here, not buried in a code comment.
- Verification: the exact commands run and their results (`bun run doctor`, `bun run build`, and, when applicable, `bun run test:e2e` and `bun run react-doctor`), not a checklist of planned tests and not a vague "tested manually."
- Screenshot: whenever the change touches UI, attach one. A full `/visual-recap` is optional; reach for it only when the change is large enough to earn the extra artifact.
