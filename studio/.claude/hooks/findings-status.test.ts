import { describe, expect, test } from 'bun:test'

import { judgmentBlocker } from './findings-status.ts'

const SCOPE = '0123456789abcdef'
const OTHER_SCOPE = 'fedcba9876543210'

describe('judgment gate', () => {
  test('clears the gate on PASS for the current scope', () => {
    expect(judgmentBlocker(`# review-judgment: PASS scope=${SCOPE}`, SCOPE)).toBeNull()
  })

  test('blocks when the findings file is missing', () => {
    expect(judgmentBlocker(null, SCOPE)).toContain('No review-judgment findings file')
  })

  test('blocks on FAIL', () => {
    expect(judgmentBlocker(`# review-judgment: FAIL scope=${SCOPE}`, SCOPE)).toContain('FAIL')
  })

  test('blocks when the reviewed scope no longer matches the code', () => {
    const blocker = judgmentBlocker(`# review-judgment: PASS scope=${OTHER_SCOPE}`, SCOPE)

    expect(blocker).toContain(OTHER_SCOPE)
    expect(blocker).toContain(SCOPE)
  })

  test('blocks a verdict line carrying no scope', () => {
    expect(judgmentBlocker('# review-judgment: PASS', SCOPE)).toContain('scope=<hash>')
  })

  test('blocks a verdict line written for a different reviewer', () => {
    expect(judgmentBlocker(`# review-types: PASS scope=${SCOPE}`, SCOPE)).toContain('review-types')
  })

  test('blocks an empty first line', () => {
    expect(judgmentBlocker('', SCOPE)).not.toBeNull()
  })
})
