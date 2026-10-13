# Phase 1: Requirements

Before writing any plan, interrogate the ask:

1. Run `/grilling` (or reason through the same questions yourself) to stress-test the request: what's the actual goal, what's out of scope, what would make this fail review.
2. Invoke `/read-the-damn-docs` for anything touching a third-party surface: TanStack Router/Query/Table, shadcn, Zod, react-hook-form, Playwright, oxlint/oxfmt. Never assume behavior from memory for these.
3. State the assumptions you're making explicitly in the plan (phase 2), not silently in code.

Skip the grill only for genuinely mechanical changes (typo fix, dependency bump with no API change). Say so instead of running the phase.
