// Integration tests for the hook's actual process wiring (exit codes,
// fail-closed behavior), not just the pure logic it delegates to.

import { describe, expect, test } from 'bun:test'
import { spawnSync } from 'node:child_process'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const HOOK_PATH = join(import.meta.dir, 'require-doctor.ts')

function makeCleanRepo(): string {
  const dir = mkdtempSync(join(tmpdir(), 'require-doctor-'))
  spawnSync('git', ['init'], { cwd: dir })
  return dir
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

describe('require-doctor hook process', () => {
  test('exits 0 immediately when stop_hook_active is true, without touching the filesystem', () => {
    const result = runHook({ stop_hook_active: true }, undefined)

    expect(result.status).toBe(0)
  })

  test('fails closed with exit 2 when CLAUDE_PROJECT_DIR is unset', () => {
    const result = runHook({ stop_hook_active: false }, undefined)

    expect(result.status).toBe(2)
    expect(result.stderr).toContain('CLAUDE_PROJECT_DIR')
  })

  test('exits 0 on a clean working tree without running doctor', () => {
    const repo = makeCleanRepo()
    try {
      const result = runHook({ stop_hook_active: false }, repo)

      expect(result.status).toBe(0)
      expect(result.stderr).toBe('')
    } finally {
      rmSync(repo, { recursive: true, force: true })
    }
  })
})
