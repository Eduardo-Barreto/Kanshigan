# Phase 5: Gates

Run, in this order, stopping at the first failure:

1. `bun run doctor`: oxlint (type-aware rules plus this project's own convention rules from `.oxlint-plugins/`), oxfmt check, typecheck, knip, unit tests. This is the whole deterministic gate and it finishes in about three seconds.
2. `bun run build`: the production bundle. `doctor`'s `tsc -b` catches type errors but not a build that fails for other reasons (a broken import, an asset Vite can't resolve). CI runs this too, so the point of running it here is catching it before the push, not instead of CI.
3. `bun run policy`: diff-level checks the per-file linters can't see (source changed with no test, dead relative link in prose, lint config widened against the base). A `notice:` line doesn't fail the run, but it does have to be answered in the PR description.
4. `bun run test:e2e`: only when the change touches routing, forms, or anything Playwright covers. This includes the `@axe-core/playwright` sweep, which is where real contrast and focus order get checked.
5. `bun run react-doctor`: only when the change touches React components. CI runs this on the PR regardless, but catching it here is cheaper than a review round-trip.

A `Stop` hook already runs `bun run doctor` and blocks you from ending the turn while it fails and the working tree has uncommitted changes. Treat the hook's block as a to-do list, not an obstacle: read the failure, fix it, let the hook re-run. It only fires on uncommitted changes, so committing and ending the turn in the same breath skips it: don't treat it as the only check.

Never bypass a gate by loosening `.oxlintrc.json`, disabling a rule inline, or skipping a test to make it pass. Fix the underlying issue. `bun run policy` compares the lint and knip config against the base branch and prints a `notice:` for anything widened, which doesn't fail the run but does have to be answered in the PR description.
