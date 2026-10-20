# CLAUDE.md

Kanshigan Studio: the annotation workbench for the Kanshigan dataset. A Bun + Hono server over the repository's `data/` tree and a React 19 front end, run on demand on the machine that holds the data and the GPU. It lives in `studio/` inside the Kanshigan repository; run every command from this folder.

## Commands

- `bun run dev`: the API with reload (`dev:api`) and Vite (`dev:web`) together; Vite proxies `/api` to the server.
- `bun run start`: production build, then the server on `127.0.0.1:4310` serving both the API and the built front end.
- `bun run tailnet` / `bun run funnel` / `bun run expose:off`: put the running server on the tailnet or the public internet through Tailscale, and take it off. See `scripts/expose.ts`.
- `bun run doctor`: oxlint (type-aware plus the project's own rules), oxfmt check, typecheck, knip, unit tests. Run before every commit.
- `bun run policy`: diff-level checks against the base branch (source changed with no test, dead relative link, lint config widened). Runs in CI and in the `gh pr create` gate.
- `bun run react-doctor`: React-specific health scan (anti-patterns, perf, a11y) layered on oxlint. Vendored `ui/` and generated files are excluded via `doctor.config.json`; the CI action reports only PR-introduced issues.
- `bun run test`: unit tests (bun test, over `src`, `.claude/hooks`, `scripts` and `.oxlint-plugins`).
- `bun run test:e2e`: Playwright end-to-end against a throwaway copy of `e2e/fixtures/library`, including the axe accessibility sweep. It builds first.
- `bun run build`: typecheck then production bundle.
- `bunx shadcn@latest add <component>`: pull a new shadcn component.

## Delivery

`/deliver` runs the full structured pipeline for this repo: grilling + docs, visual plan with test plan, implementation with an anti-slop cleanup pass, the deterministic gate, one Opus review of intent and cross-file logic, PR, then babysitting CI and the adversarial review workflow to green. It overrides the generic `/deliver`. See `.claude/skills/deliver/SKILL.md`.

Three hooks in `.claude/settings.json` enforce this regardless of which flow you use: a `Stop` hook runs `bun run doctor` whenever the working tree has uncommitted changes, a `PreToolUse` hook blocks `gh pr create` until the gate passes and `review-judgment` has signed off on the current scope hash, and a `PostToolUse` hook lints each edited file and feeds the diagnostics straight back.

Everything a parser can decide is enforced by `bun run doctor`, not by a reviewer reading this file. The conventions below are checked by oxlint's type-aware rules plus this project's own rules in `.oxlint-plugins/`, each with a test in `.oxlint-plugins/project-conventions.test.ts`. If you disagree with a rule, change the rule; don't work around it.

## Stack

The server is Hono on Bun (`src/server`). Routes are chained in `src/server/app.ts` so `hc<StudioApi>` in `src/lib/api.ts` derives request and response types from them. Every request body and path parameter goes through a Zod schema from `src/shared`, and the client parses every response with the same schemas.

There is no database. The server reads and writes the annotation folders the training and evaluation scripts already use (`data/annotations/<dataset>/{images,labels}`), plus one sidecar JSON per clip for what a YOLO label cannot hold. `README.md` documents the layout. Never write a label file outside `src/server/clip-write.ts`: it keeps the image placement, the label and the sidecar consistent.

Long work (YouTube download, round detection, cut + SAM 3) runs as jobs in `src/server/jobs.ts`, one at a time, by spawning the Python scripts in `../experiments`. Commands are argument arrays, never shell strings.

Routing is TanStack Router, file-based in `src/routes`, with `autoCodeSplitting` on. `routeTree.gen.ts` is generated, so never edit it; it stays committed so typecheck works without a dev server. Everything behind the session guard lives under the pathless `_app` layout.

UI is shadcn/ui on Tailwind v4. Components in `src/components/ui` are vendored, so don't hand-edit them unless you mean to. The visual language (neutral dark surface, windows with a title bar, mono tabular numbers, chips) lives in `src/index.css` and `src/components/window.tsx`.

Data goes through TanStack Query with `useSuspenseQuery` and the option factories in `src/lib/api.ts`. The annotation workbench is the exception: `src/features/workbench/workbench-store.ts` owns the editing session as an external store, because edits, playback and background saves are imperative.

Forms use react-hook-form and Zod (`zodResolver`).

## Structure

```
src/
  server/            Hono API: clips, ingest jobs, auth, exports
  shared/            Zod schemas and pure logic used by server and client
  routes/            file-based routes (__root, login, _app layout and its children)
  features/          library, workbench, ingest, auth
  components/        window primitives, brand mark, status bar
  components/ui/     shadcn, vendored, excluded from knip and most lint rules
  lib/               typed API client, query client, formatting
  test/              bun test setup + jest-dom matcher types
e2e/                 Playwright specs and the fixture library they run against
scripts/             expose (Tailscale), e2e-server, scope-hash, diff-policy
.oxlint-plugins/     this project's conventions as real lint rules
.claude/             hooks, review agents, and the /deliver skill
```

CI for this folder is `../.github/workflows/studio.yml`.

## Adding a route

Create `src/routes/<name>.tsx` exporting `Route = createFileRoute('/<name>')({ component })`. The plugin regenerates `routeTree.gen.ts` on the next dev or build. Link with the typed `<Link to="/<name>">`; wrong paths fail typecheck.

## Workflow

IMPORTANT: every change ships as a pull request, and you babysit that PR until it is green.

1. Read the docs first. Run `/read-the-damn-docs` for any external API, library, framework, or CLI the task touches. Ground the work in current official docs, not memory.
2. For a complex or multi-file task, propose a `/visual-plan` and wait for the user to accept or decline before implementing. Skip it for small, obvious changes.
3. Work on a branch. Never commit to `main`.
4. Prefer the best-practice solution. Avoid workarounds and hardcoded values; if one is unavoidable, say so in the PR with the reason and the drawback.
5. Run `bun run doctor` until it is clean. Write a test for every bug you fix.
6. Run `/improve` before opening the PR and fold in what is worth doing now.
7. Open the PR with a description that explains what changed and how (see Pull requests).
8. Babysit the PR: wait for CI and the adversarial review, read every failure and comment, fix, push, and repeat until CI is green and no 🔴 finding is open.

YOU MUST NOT call a task done until `bun run doctor` passes, relevant e2e passes, CI is green, and the adversarial review is resolved. No "should work", no skipped checks, no `.only`/`.skip`, no TODO placeholders.

## Pull requests

Lead with what changed and why. Cover the design decisions, the acceptance criteria, and the exact commands you ran to verify. Document every drawback and any decision you made without explicit approval, especially a workaround or a hardcoded value, in the PR body or a PR comment. Verification lives in the description so a reviewer can rerun it.

## Conventions

The global `~/.claude/CLAUDE.md` rules apply. Project-specific notes:

- oxlint and oxfmt own lint and format. Config in `.oxlintrc.json` and `.oxfmtrc.json`: single quotes, no semicolons, width 100, sorted imports. `bun run doctor` is the gate.
- Named exports only (`import/no-default-export` is an error). Config files and the oxlint plugin entrypoint are the only exceptions.
- `any` is an error. `console` is a warning outside `scripts/` and `.claude/`. Unused vars must be prefixed `_` or deleted.
- Accessibility is enforced (`jsx-a11y` statically, `@axe-core/playwright` at runtime). Use accessible roles and labels, and query by them in tests.
- No narrating comments. If you reach for one to explain how code works, extract a named function or variable instead. The only comments worth keeping are docstrings (JSDoc/doxygen) and the rare load-bearing "why" that prevents a maintainer from breaking something. Exported functions need a TSDoc block; that one is linted.
- Comments are written in English, even though PR prose is pt-BR. `project/comment-must-be-english` enforces it.
- A `TODO` or `workaround` comment needs a tracked issue link. `project/workaround-needs-issue-link` enforces it.
- Commits follow Conventional Commits, in English, single-line: `type(scope): description`.
- Never add tool attribution to a commit or PR: no "Claude", no co-authored-by trailer, no session footer.
- Never bypass git hooks with `--no-verify`; let lefthook run.

## Testing

Test behavior, not implementation. Query by role and label, never by class or test-id (`project/no-test-id-query` enforces it). Unit tests use bun test with happy-dom and Testing Library; e2e uses Playwright with accessible locators. Server tests build a temporary data directory and must not need ffmpeg, ffprobe or Python, so they run in CI. Write a test for every bug you fix: `bun run policy` fails a diff that changes `src/` without touching a test.

## Writing

Every piece of text this repo produces follows `docs/anti-slop-guidelines.md`: PR bodies, commit messages, code comments, GitHub comments, and UI copy above all. That file is a versioned copy of https://files.barreto.sh/slop.md; read the local copy, and when the upstream changes, bring the change here in the same PR. Lead with the result, cut filler, stay specific.

PR descriptions, PR comments, and code review are written in Brazilian Portuguese (pt-BR). Commit messages and code comments stay in English.
