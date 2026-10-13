// Pure verdict parsing for the one phase-6 reviewer, split out from require-review.ts so
// it's importable and testable without touching the filesystem.

import { join } from 'node:path'

import { HASH_LENGTH } from '../../scripts/scope-hash.ts'

export const FINDINGS_DIR = join('.omc', 'state', 'deliver', 'findings')
export const JUDGMENT_REVIEWER = 'review-judgment'

// Built from HASH_LENGTH rather than hardcoding 16, so the hash and the line that carries it
// can't drift into a gate that blocks forever.
const VERDICT_LINE = new RegExp(
  String.raw`^#\s*review-judgment:\s*(?<verdict>PASS|FAIL)\s+scope=(?<scope>[0-9a-f]{${HASH_LENGTH}})\s*$`,
)

/**
 * Decides whether the reviewer's findings file clears the gate.
 * @param firstLine - first line of the findings file, or null when the file is absent
 * @param expectedScope - hash of the code as it stands now, from scripts/scope-hash.ts
 * @returns null when the gate is satisfied, otherwise the reason to block, phrased for the agent
 */
export function judgmentBlocker(firstLine: string | null, expectedScope: string): string | null {
  if (firstLine === null) {
    return `No ${JUDGMENT_REVIEWER} findings file. Spawn the reviewer, then retry.`
  }

  const match = VERDICT_LINE.exec(firstLine)
  if (match?.groups === undefined) {
    return `${JUDGMENT_REVIEWER} findings file doesn't start with '# ${JUDGMENT_REVIEWER}: PASS scope=<hash>'. Got: '${firstLine}'.`
  }

  const { verdict, scope } = match.groups
  if (verdict === 'FAIL') {
    return `${JUDGMENT_REVIEWER} reported FAIL. Fix the findings it listed, then re-run it.`
  }
  if (scope !== expectedScope) {
    return `${JUDGMENT_REVIEWER} reviewed scope ${scope}, but the code now hashes to ${expectedScope}. Re-run the reviewer against the current diff.`
  }
  return null
}
