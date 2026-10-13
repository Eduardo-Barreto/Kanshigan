# Phase 2: Plan

The plan is one artifact covering both product and implementation, not two separate plans:

1. Draft it with `/visual-plan`: product and UX (wireframes or annotated mockups when UI is affected) together with implementation (file map, architecture, which `src/routes/` and `src/features/` change, TanStack Query keys, forms) in the same document, plus an explicit test plan section.
2. Before showing it to the user, spawn `review-plan` against the draft. It catches a bad plan while it's still cheap to fix, not after code has been written against it. Revise and re-run `review-plan` until it passes.
3. Present the reviewed plan to the user and iterate on it for as many rounds as it takes.
4. Once the user gives explicit final sign-off, move to phase 3 (`/compact`).

Test plan requirements, regardless of what else the plan covers:

- Which behaviors get a unit test (bun test, Testing Library, query by role/label)
- Which need Playwright e2e
- Which bug, if any, is being prevented from recurring
