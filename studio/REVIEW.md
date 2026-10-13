# REVIEW.md

Tuning for the adversarial PR review in `.github/workflows/adversarial-review.yml`. Both review lanes read this file and treat it as the highest-priority instruction.

It also sets the bar for the local `review-judgment` agent, which reviews the same kind of diff in phase 6 of `/deliver`, before the PR exists.

## Severity for this repo

🔴 **Important** (fix before merge): a logic bug that produces wrong results; a `jsx-a11y` accessibility violation; type-unsafety introduced by `any` or `as`; anything that breaks the build or a CI check; UI copy that violates `docs/anti-slop-guidelines.md`.

🟡 **Nit** (worth fixing, not blocking): everything else, including style and naming. Post at most five; summarize the rest as a count.

🟣 **Pre-existing** (not introduced by this PR): report at most as a note, never as a blocker.

## Evidence bar

A behavior claim needs a `file:line` citation in the source, not an inference from naming. A 🔴 or 🟡 verdict needs actual command output (a targeted test run, `grep`, or `bun run typecheck`), not reasoning alone. A finding backed only by reasoning caps at confidence 50 and cannot be 🔴.

## Do not flag

Pre-existing issues, anything a linter already catches, pedantic nits a senior engineer would wave through, code that looks like a bug but is correct, and issues silenced on purpose in code (a lint-ignore comment). If you are not certain a finding is real, drop it. False positives waste the author's time.

## Re-review convergence

After the first review, suppress new nits and post 🔴 Important findings only, so a one-line fix does not trigger a fresh round of style comments.

## Summary shape

Open the review summary with a one-line tally such as `2 important, 3 nits`, and lead with "no important findings" when that is the case.
