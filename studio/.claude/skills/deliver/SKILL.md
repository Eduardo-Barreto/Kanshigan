---
name: deliver
description: 'Full-cycle delivery pipeline for this repo: grounded requirements interrogation, visual plan with test plan, implementation with a bounded anti-slop cleanup pass, a deterministic quality gate, one LLM review of intent and cross-file logic, PR, then babysitting CI and the repo's own adversarial review workflow to green. Overrides the generic /deliver for this project.'
disable-model-invocation: true
argument-hint: [issue-number-or-description]
allowed-tools: Bash(bun scripts/scope-hash.ts:*)
---

# Deliver

Run every non-trivial feature, fix, or refactor through these phases in order. Each phase links to its reference file; read it when you reach that phase, not before.

1. **Requirements**: interrogate the ask and ground it in real docs. See [reference/requirements.md](reference/requirements.md).
2. **Plan**: one visual plan covering product and implementation, with an explicit test plan, reviewed by `review-plan` before you see it, iterated with the user until signed off. See [reference/plan.md](reference/plan.md).
3. **Compact**: run `/compact` once the plan is signed off. Claude Code's own compaction keeps a summary of what was approved; there's no separate file to write here.
4. **Implementation**: build against the approved plan, then a bounded `ai-slop-cleaner` pass on the touched files. See [reference/implementation.md](reference/implementation.md).
5. **Gates**: the deterministic pass, in about three seconds. See [reference/gates.md](reference/gates.md). A `Stop` hook already enforces `bun run doctor` before you can end the turn, so this phase is a checkpoint, not the only line of defense.
6. **Adversarial review**: one Opus reviewer, judging only what a parser can't. See [reference/adversarial-review.md](reference/adversarial-review.md).
7. **PR**: open it once every phase above is clean; this is where the "why" gets written down, once, for whoever reads it next. See [reference/pr.md](reference/pr.md).
8. **Babysit**: watch CI and the adversarial review workflow to green, loop fixes back through the gates. See [reference/babysit.md](reference/babysit.md).

Don't skip a phase because the change looks small. A phase that genuinely doesn't apply (no UI touched, so no visual plan needed) gets a one-line note explaining why, not silence.
