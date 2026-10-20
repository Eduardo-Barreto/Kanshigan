#!/usr/bin/env bun
/**
 * Content hash of everything a reviewer needs to look at: every file this branch changed
 * against the default branch, plus every untracked file, hashed by path and current content.
 *
 * Hashing content rather than diff text is what makes this indifferent to committing,
 * amending, rebasing or staging. A review stays fresh until the code itself changes, which
 * is why the old commit-timestamp gate needed three paragraphs of "don't commit mid-review".
 */

import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { existsSync, readFileSync } from 'node:fs'

const GIT_TIMEOUT_MS = 10_000

/** Hex characters in a scope hash. The findings-file verdict line is validated against this. */
export const HASH_LENGTH = 16

function git(args: string[]): string {
  const result = spawnSync('git', args, { encoding: 'utf8', timeout: GIT_TIMEOUT_MS })
  if (result.status !== 0) {
    // ||, not ??: a timeout kill returns stderr as an empty string, not
    // null/undefined, so ?? would swallow the real reason.
    throw new Error(`git ${args.join(' ')} failed: ${result.stderr || String(result.error)}`)
  }
  return result.stdout
}

function gitList(args: string[]): string[] {
  return git(args).split('\n').filter(Boolean)
}

/** Resolves the default branch from origin's HEAD, falling back to main in a clone without it. */
export function defaultBranch(): string {
  const ref = spawnSync('git', ['symbolic-ref', 'refs/remotes/origin/HEAD'], {
    encoding: 'utf8',
    timeout: GIT_TIMEOUT_MS,
  })
  if (ref.status === 0 && ref.stdout.trim()) {
    return ref.stdout.trim().replace('refs/remotes/origin/', '')
  }
  return 'main'
}

/**
 * Resolves the merge-base of `base` and HEAD, preferring the remote-tracking ref and falling
 * back to a local branch of the same name.
 *
 * Remote-tracking first because that is what the PR will actually be compared against. A local
 * branch of the same name is a mirror that goes stale the moment someone else merges, and
 * every consumer of this scope reads too much when it does: `bun run policy` reports widening
 * that is already on the base branch, and `review-judgment` spends its one pass on code the
 * PR never touched.
 * @throws when neither ref resolves, so a caller can't hash a scope it failed to determine
 */
export function mergeBase(base: string): string {
  for (const ref of [`origin/${base}`, base]) {
    const result = spawnSync('git', ['merge-base', ref, 'HEAD'], {
      encoding: 'utf8',
      timeout: GIT_TIMEOUT_MS,
    })
    if (result.status === 0) return result.stdout.trim()
  }
  throw new Error(`git merge-base failed against both 'origin/${base}' and '${base}'.`)
}

/** Every path in the review scope, whether the change is committed, staged or untracked. */
export function scopeFiles(): string[] {
  const base = mergeBase(defaultBranch())
  return [
    ...new Set([
      // --relative keeps the scope to this package when it lives in a subfolder of a larger repo.
      ...gitList(['diff', '--name-only', '--relative', base]),
      ...gitList(['ls-files', '--others', '--exclude-standard']),
    ]),
  ].toSorted()
}

/** Hashes path plus current content for every file in scope. A deleted file hashes as empty. */
export function scopeHash(): string {
  const digest = createHash('sha256')
  for (const file of scopeFiles()) {
    digest.update(file)
    digest.update('\0')
    digest.update(existsSync(file) ? readFileSync(file) : '')
    digest.update('\0')
  }
  return digest.digest('hex').slice(0, HASH_LENGTH)
}

if (import.meta.main) {
  console.log(scopeHash())
}
