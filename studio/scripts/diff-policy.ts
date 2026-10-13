#!/usr/bin/env bun
/**
 * Diff-level policy the per-file linters can't see: source changed without a test, a dead
 * relative link in prose, and lint config widened against the base branch.
 *
 * Widening is reported, not blocked. Whether a new `"off"` is justified is a judgment call,
 * which is `review-judgment`'s job; this only makes sure the call is visible instead of
 * buried in a config diff nobody reads.
 */

import { spawnSync } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'
import { dirname, join, normalize } from 'node:path'

import { defaultBranch, mergeBase, scopeFiles } from './scope-hash.ts'

const GIT_TIMEOUT_MS = 10_000
const SOURCE_FILE = /^src\/.*\.tsx?$/
const GENERATED_FILE = /routeTree\.gen\.ts$/
const TEST_FILE = /\.(?:test|spec)\.tsx?$/
const MARKDOWN_FILE = /\.md$/
// The lookahead rejects a protocol (`https:`, `mailto:`), a root-absolute path and a bare
// anchor. Everything else is repo-relative, including the bare form this repo actually uses
// (`[reference/plan.md](reference/plan.md)`), not just `./` and `../`.
const RELATIVE_LINK = /\[[^\]]*\]\((?!\w+:|\/|#)(?<target>[^)\s#]+)/g
const SUPPRESSION_CONFIGS = ['.oxlintrc.json', 'knip.json', 'doctor.config.json']

/** Reports source files changed with no accompanying test change. */
export function missingTestCoverage(changed: string[]): string[] {
  const source = changed.filter(
    (file) => SOURCE_FILE.test(file) && !GENERATED_FILE.test(file) && !TEST_FILE.test(file),
  )
  if (source.length === 0) return []
  if (changed.some((file) => TEST_FILE.test(file))) return []
  return source
}

/** Resolves every relative markdown link in `files` and reports the ones that point nowhere. */
export function deadRelativeLinks(
  files: { path: string; content: string }[],
  exists: (path: string) => boolean,
): string[] {
  const dead: string[] = []
  for (const file of files) {
    for (const match of file.content.matchAll(RELATIVE_LINK)) {
      const target = match.groups?.target
      if (target === undefined) continue
      const resolved = normalize(join(dirname(file.path), target))
      if (!exists(resolved)) dead.push(`${file.path} -> ${target}`)
    }
  }
  return dead
}

/**
 * Counts how many rules a config turns off and how many paths it ignores.
 * @returns a single number, so a caller can compare two revisions of the same file
 */
export function suppressionCount(config: unknown): number {
  if (Array.isArray(config)) {
    return config.reduce((total: number, entry) => total + suppressionCount(entry), 0)
  }
  if (typeof config !== 'object' || config === null) return config === 'off' ? 1 : 0

  let total = 0
  for (const [key, value] of Object.entries(config)) {
    if (/ignore/i.test(key) && Array.isArray(value)) {
      total += value.length
      continue
    }
    total += suppressionCount(value)
  }
  return total
}

function fileAtRevision(revision: string, path: string): string | null {
  const result = spawnSync('git', ['show', `${revision}:${path}`], {
    encoding: 'utf8',
    timeout: GIT_TIMEOUT_MS,
  })
  return result.status === 0 ? result.stdout : null
}

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text)
  } catch {
    return null
  }
}

/** Reports each suppression config that turns off more than it did on the base branch. */
export function widenedConfigs(base: string): string[] {
  const widened: string[] = []
  for (const path of SUPPRESSION_CONFIGS) {
    if (!existsSync(path)) continue
    const before = fileAtRevision(base, path)
    if (before === null) continue
    const beforeCount = suppressionCount(parseJson(before))
    const afterCount = suppressionCount(parseJson(readFileSync(path, 'utf8')))
    if (afterCount > beforeCount) {
      widened.push(`${path}: ${beforeCount} -> ${afterCount} suppressions`)
    }
  }
  return widened
}

if (import.meta.main) {
  const base = mergeBase(defaultBranch())
  const changed = scopeFiles()

  const untested = missingTestCoverage(changed)
  const dead = deadRelativeLinks(
    changed
      .filter((file) => MARKDOWN_FILE.test(file) && existsSync(file))
      .map((file) => ({ path: file, content: readFileSync(file, 'utf8') })),
    existsSync,
  )
  const widened = widenedConfigs(base)

  for (const entry of widened) {
    console.log(`notice: config widened, justify it in the PR: ${entry}`)
  }

  const failures = [
    ...untested.map((file) => `source changed with no test change: ${file}`),
    ...dead.map((link) => `dead relative link: ${link}`),
  ]
  for (const failure of failures) {
    console.error(`error: ${failure}`)
  }
  process.exit(failures.length > 0 ? 1 : 0)
}
