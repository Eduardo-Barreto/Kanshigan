#!/usr/bin/env bun
// Blocks Claude from stopping while the working tree has uncommitted changes
// and `bun run doctor` (oxlint, oxfmt check, typecheck, knip, unit tests) fails.
// console.error here is the hook's protocol with Claude Code: stderr
// becomes the block reason. See the `.claude/**` override for
// `eslint/no-console` in .oxlintrc.json.

import { spawnSync } from 'node:child_process'

import { readHookPayload, requireProjectDir, runHookGuard } from './hook-input.ts'

if (readHookPayload()?.stop_hook_active === true) process.exit(0)

await runHookGuard(() => {
  requireProjectDir('finish without checking bun run doctor', 2)

  // No path restriction: `bun run doctor` runs oxfmt/oxlint over the whole
  // repo, not just src/e2e, so the trigger has to match that scope or a
  // .claude/-only change can end its turn with doctor actually failing.
  // A timeout kill sets status to null, which each check below already
  // treats as failure, so a hung process can't stall this indefinitely.
  const GIT_TIMEOUT_MS = 10_000
  const gitStatus = spawnSync('git', ['status', '--porcelain', '--untracked-files=all'], {
    encoding: 'utf8',
    timeout: GIT_TIMEOUT_MS,
  })
  if (gitStatus.status !== 0) {
    throw new Error(
      // ||, not ??: a timeout kill returns stderr as an empty string, not
      // null/undefined, so ?? would silently swallow the real reason.
      `git status failed: ${gitStatus.stderr || String(gitStatus.error)}. Can't tell if there's anything to check; fix the git error before finishing.`,
    )
  }
  if (!gitStatus.stdout.trim()) {
    process.exit(0)
  }

  const DOCTOR_OUTPUT_TAIL_LINES = 20
  // Comfortably under this hook's own 300s Stop-hook timeout in settings.json.
  const DOCTOR_TIMEOUT_MS = 120_000

  const doctor = spawnSync('bun', ['run', 'doctor'], {
    encoding: 'utf8',
    timeout: DOCTOR_TIMEOUT_MS,
  })
  if (doctor.status !== 0) {
    // Both streams: oxlint and tsc report diagnostics on stderr, so a
    // stdout-only tail comes back empty on exactly the failures worth quoting.
    // Bun's process.exit can truncate a large pending pipe write, so the full
    // output is never echoed; this tail is the entire report Claude gets.
    const tail = `${doctor.stdout}${doctor.stderr}`
      .trim()
      .split('\n')
      .slice(-DOCTOR_OUTPUT_TAIL_LINES)
      .join('\n')
    const reason = tail || String(doctor.error ?? 'bun run doctor exited with no output')
    throw new Error(
      `bun run doctor failed. Fix the reported lint/format/typecheck/knip/test errors before finishing.\n\n${reason}`,
    )
  }
})
