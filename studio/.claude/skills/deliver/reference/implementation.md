# Phase 4: Implementation

Build directly against the plan approved and compacted in phases 2-3.

- Follow `CLAUDE.md` conventions: named exports, no default exports, strict types, no `any`/`as`, early return, react-hook-form + Zod for forms, TanStack Query for data.
- Commit in atomic, reviewable chunks as you go, not as one giant diff at the end. This makes phase 6 (adversarial review) tractable and phase 7 (PR) honest.
- If implementation reveals the plan was wrong, stop and go back to phase 2. Don't quietly reinterpret it to fit what you've already built.

Once the implementation itself is done, before moving to phase 5, run `ai-slop-cleaner` (standard mode, not `--review`) bounded to the files this delivery touched: dead code, duplicate logic, needless wrappers, weak regression coverage. It edits code and re-verifies its own regression tests as part of its own workflow; phase 5's gates and phase 6's reviewers still independently judge the result afterward, so this cleanup pass doesn't skip either.
