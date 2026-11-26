#!/usr/bin/env bun
/**
 * The diff a PR is measured by: every file this branch changed against the default branch,
 * plus every untracked file.
 */

import { spawnSync } from 'node:child_process'

const GIT_TIMEOUT_MS = 10_000

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
 * `bun run policy` then reports widening that is already on the base branch.
 * @throws when neither ref resolves, so a caller can't measure a scope it failed to determine
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

/** Every path in the diff scope, whether the change is committed, staged or untracked. */
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
