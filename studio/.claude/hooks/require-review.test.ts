// Integration tests for the hook's actual process wiring (exit codes, fail-closed
// behavior), not the pure logic it delegates to (covered in pr-create-matcher.test.ts
// and findings-status.test.ts).

import { describe, expect, test } from 'bun:test'
import { spawnSync } from 'node:child_process'
import { cpSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { FINDINGS_DIR, JUDGMENT_REVIEWER } from './findings-status.ts'

const HOOK_PATH = join(import.meta.dir, 'require-review.ts')
const SCOPE_HASH_PATH = join(import.meta.dir, '..', '..', 'scripts', 'scope-hash.ts')

function makeRepoWithoutFindings(): string {
  const dir = mkdtempSync(join(tmpdir(), 'require-review-'))
  spawnSync('git', ['init', '-b', 'main'], { cwd: dir })
  // `.omc` has to be ignored, exactly as it is in this repo's .gitignore: the findings file
  // lives under it, and an untracked findings file joins the very scope it certifies, so
  // writing the verdict would invalidate the verdict.
  writeFileSync(join(dir, '.gitignore'), '.omc\n')
  spawnSync('git', ['add', '.gitignore'], { cwd: dir })
  spawnSync('git', ['-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-m', 'init'], {
    cwd: dir,
  })
  return dir
}

/** Writes a PASS verdict carrying the hash the reviewer's own CLI reports for this repo. */
function writePassingFindings(repo: string): void {
  const hash = spawnSync('bun', [SCOPE_HASH_PATH], { cwd: repo, encoding: 'utf8' }).stdout.trim()
  const dir = join(repo, FINDINGS_DIR)
  mkdirSync(dir, { recursive: true })
  writeFileSync(
    join(dir, `${JUDGMENT_REVIEWER}.md`),
    `# ${JUDGMENT_REVIEWER}: PASS scope=${hash}\n\nNo findings.\n`,
  )
}

function runHook(payload: unknown, projectDir: string | undefined) {
  const env = { ...process.env }
  if (projectDir === undefined) {
    delete env.CLAUDE_PROJECT_DIR
  } else {
    env.CLAUDE_PROJECT_DIR = projectDir
  }
  return spawnSync('bun', [HOOK_PATH], { input: JSON.stringify(payload), encoding: 'utf8', env })
}

describe('require-review hook process', () => {
  test('exits 0 for a command that is not gh pr create, without touching the filesystem', () => {
    const result = runHook({ tool_input: { command: 'ls -la' } }, undefined)

    expect(result.status).toBe(0)
  })

  test('fails closed with exit 2 when CLAUDE_PROJECT_DIR is unset for a gh pr create command', () => {
    const result = runHook({ tool_input: { command: 'gh pr create --title x' } }, undefined)

    expect(result.status).toBe(2)
    expect(result.stderr).toContain('CLAUDE_PROJECT_DIR')
  })

  test('blocks with exit 2 when the reviewer has written no findings file', () => {
    const repo = makeRepoWithoutFindings()
    try {
      const result = runHook({ tool_input: { command: 'gh pr create --title x' } }, repo)

      expect(result.status).toBe(2)
      expect(result.stderr).toContain('review-judgment')
    } finally {
      rmSync(repo, { recursive: true, force: true })
    }
  })

  test('falls through to the gate when the payload carries no command', () => {
    const result = runHook({ tool_input: {} }, undefined)

    expect(result.status).toBe(2)
    expect(result.stderr).toContain('CLAUDE_PROJECT_DIR')
  })

  // Regression: scripts/init.ts used to delete `scripts/` wholesale, which took scope-hash.ts
  // with it. As a static import it threw during module load, before runHookGuard existed to
  // catch it, so the process exited 1, which Claude Code reads as non-blocking. Every
  // project made from this template shipped with an open PR gate that still looked
  // configured. Exit 2 is the whole point of the assertion.
  test('fails closed with exit 2 when scope-hash.ts cannot be resolved', () => {
    const dir = mkdtempSync(join(tmpdir(), 'require-review-noscripts-'))
    try {
      const hooks = join(dir, '.claude', 'hooks')
      mkdirSync(hooks, { recursive: true })
      cpSync(import.meta.dir, hooks, { recursive: true })

      const result = spawnSync('bun', [join(hooks, 'require-review.ts')], {
        input: JSON.stringify({ tool_input: { command: 'gh pr create --title x' } }),
        encoding: 'utf8',
        env: { ...process.env, CLAUDE_PROJECT_DIR: dir },
      })

      expect(result.status).toBe(2)
      expect(result.stderr).toContain('scope-hash.ts')
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })

  // The reviewer computes its hash by running scripts/scope-hash.ts, while the hook computes
  // one in-process. This is the only case that proves the two agree; if they ever diverge, the
  // gate blocks a review that is genuinely fresh and no other test notices.
  test('accepts a PASS verdict carrying the hash the reviewer CLI reports', () => {
    const repo = makeRepoWithoutFindings()
    try {
      writePassingFindings(repo)
      const result = runHook({ tool_input: { command: 'gh pr create --title x' } }, repo)

      expect(result.stderr).not.toContain(JUDGMENT_REVIEWER)
    } finally {
      rmSync(repo, { recursive: true, force: true })
    }
  })
})
