// Integration tests for the hook's real wiring: it must surface diagnostics as context
// and stay silent (and non-blocking) on every other path.

import { describe, expect, test } from 'bun:test'
import { spawnSync } from 'node:child_process'
import { mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

const HOOK_PATH = join(import.meta.dir, 'lint-touched.ts')
const PROJECT_DIR = join(import.meta.dir, '..', '..')
// Not `src/`: knip scans it and lefthook runs knip and the tests in parallel, so a probe file
// living there fails the push of whoever happens to race it. This directory is gitignored,
// which oxlint honours while walking a tree but not when handed an explicit path.
const PROBE_DIR = join(PROJECT_DIR, 'tmp-lint-probe')

function runHook(filePath: string | undefined) {
  const toolInput = filePath === undefined ? {} : { file_path: filePath }
  return spawnSync('bun', [HOOK_PATH], {
    input: JSON.stringify({ tool_name: 'Edit', tool_input: toolInput }),
    encoding: 'utf8',
    env: { ...process.env, CLAUDE_PROJECT_DIR: PROJECT_DIR },
  })
}

function withProbe(files: Record<string, string>, assert: (probePath: string) => void) {
  mkdirSync(PROBE_DIR, { recursive: true })
  try {
    const [firstName] = Object.keys(files)
    for (const [name, content] of Object.entries(files)) {
      writeFileSync(join(PROBE_DIR, name), content)
    }
    assert(join(PROBE_DIR, String(firstName)))
  } finally {
    rmSync(PROBE_DIR, { recursive: true, force: true })
  }
}

describe('lint-touched hook', () => {
  test('feeds oxlint diagnostics back as additional context', () => {
    withProbe({ 'vague.ts': 'export const data = 1\n' }, (probe) => {
      const result = runHook(probe)
      // Both streams in the message: this assertion failed once in CI while passing locally,
      // and a bare "expected to contain" said nothing about which half went wrong.
      const evidence = `stdout: ${result.stdout}\nstderr: ${result.stderr}`

      expect(result.status).toBe(0)
      expect(result.stdout, evidence).toContain('additionalContext')
      expect(result.stdout, evidence).toContain('no-vague-identifier')
    })
  })

  test('says so when oxlint fails, instead of reading as a clean file', () => {
    withProbe(
      {
        'clean.ts': 'export const value = 1\n',
        '.oxlintrc.json': '{ "rules": { "no-such-rule-exists": "error" } }',
      },
      (probe) => {
        const result = runHook(probe)

        expect(result.status).toBe(0)
        expect(result.stdout).toContain('tooling failure')
      },
    )
  })

  test('stays silent on a file that lints clean', () => {
    const result = runHook(join(PROJECT_DIR, 'src', 'lib', 'utils.ts'))

    expect(result.status).toBe(0)
    expect(result.stdout).toBe('')
  })

  test('skips a file oxlint does not lint', () => {
    const result = runHook(join(PROJECT_DIR, 'README.md'))

    expect(result.status).toBe(0)
    expect(result.stdout).toBe('')
  })

  test('exits without output when the payload carries no file path', () => {
    const result = runHook(undefined)

    expect(result.status).toBe(0)
    expect(result.stdout).toBe('')
  })
})
