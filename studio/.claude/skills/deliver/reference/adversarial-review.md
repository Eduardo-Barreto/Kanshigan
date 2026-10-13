# Phase 6: Adversarial review

By the time you get here, `bun run doctor` already passed, which means every mechanical check is green: type-safety and the 59 type-aware rules, complexity and nesting limits, naming, docstrings on exported functions, comment language, workaround comments without a tracked issue, test titles, test-id queries, accessibility lint, dead code, magic numbers, and this project's own conventions (`.oxlint-plugins/`: TanStack Query keys, `zodResolver` on forms, no fetch in `useEffect`, no manual `isLoading` branch, validated boundaries).

So one reviewer runs here, and it only judges what a parser can't.

Spawn `review-judgment` (Opus) against this branch's diff: `git diff <default-branch>...HEAD` plus anything uncommitted on top (`git diff HEAD` for tracked files, `git ls-files --others --exclude-standard` for untracked ones). Tell it both, not just the committed part.

It writes its verdict to `.omc/state/deliver/findings/review-judgment.md`, first line `# review-judgment: PASS scope=<hash>` or `FAIL`, where the hash comes from `bun scripts/scope-hash.ts`. Read that file back rather than relying on the subagent's returned text: that hand-off has been unreliable in practice, and the file is what the `gh pr create` gate reads too.

## Freshness

The gate compares the `scope=` hash in that file against the current one. The hash covers the content of every file in the diff, so committing, amending, rebasing or staging in the middle of a review changes nothing. Only a real code change invalidates the verdict, and then it invalidates it correctly.

That means there's no commit dance to perform: fix findings, commit whenever you like, and re-run the reviewer only when you actually changed code it had reviewed.

## If it FAILs

Fix the specific findings it listed, then re-run it once against the current state. Don't spawn extra reviewers for a second opinion: this pipeline deliberately has one judgment lane, because a panel asked to find gaps will report some whether or not they exist, and chasing those cost more than the fixes.

`.github/workflows/adversarial-review.yml` reviews this diff too, in CI, once the PR exists in phase 8. That lane is independent of this one: it has its own proposer and skeptic, it reads `REVIEW.md` for severity, and it runs on the pushed diff rather than the working tree. It costs no session tokens and runs asynchronously, so don't try to pre-empt it here.

Don't self-approve: `review-judgment` never runs in the same context that wrote the code.
