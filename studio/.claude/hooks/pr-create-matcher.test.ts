import { describe, expect, test } from 'bun:test'

import { isPrCreateCommand } from './pr-create-matcher.ts'

describe('isPrCreateCommand', () => {
  test('matches a plain gh pr create invocation', () => {
    expect(isPrCreateCommand('gh pr create --title foo')).toBe(true)
  })

  test('matches gh pr create behind a global flag', () => {
    expect(isPrCreateCommand('gh --repo owner/repo pr create')).toBe(true)
  })

  test('matches the raw REST route for creating a PR', () => {
    expect(isPrCreateCommand('gh api repos/owner/repo/pulls -X POST')).toBe(true)
    expect(isPrCreateCommand('gh api repos/owner/repo/pulls --method POST')).toBe(true)
  })

  test('does not match a read-only gh api call against the pulls route', () => {
    expect(isPrCreateCommand('gh api repos/owner/repo/pulls/123/comments')).toBe(false)
  })

  test('matches the raw REST route even when the route argument is quoted', () => {
    expect(isPrCreateCommand('gh api "repos/owner/repo/pulls" -X POST')).toBe(true)
    expect(isPrCreateCommand("gh api 'repos/owner/repo/pulls' --input body.json")).toBe(true)
  })

  test('does not match a POST to a pulls sub-resource, like replying to a review comment', () => {
    expect(
      isPrCreateCommand(
        'gh api repos/owner/repo/pulls/42/comments/7/replies -f body=fixed --method POST',
      ),
    ).toBe(false)
    expect(
      isPrCreateCommand('gh api "repos/owner/repo/pulls/42/comments/7/replies" --method POST'),
    ).toBe(false)
  })

  test('matches a pulls route call that implies POST via a field flag instead of an explicit method', () => {
    expect(isPrCreateCommand('gh api repos/owner/repo/pulls -f title=x -f head=branch')).toBe(true)
    expect(isPrCreateCommand('gh api repos/owner/repo/pulls --input body.json')).toBe(true)
  })

  test('matches gh pr create even when each token is individually quoted', () => {
    expect(isPrCreateCommand('gh "pr" "create" --title x')).toBe(true)
    expect(isPrCreateCommand('\'gh\' pr create --title "fix"')).toBe(true)
  })

  test('matches a wrapped invocation quoted as a single shell -c argument', () => {
    expect(isPrCreateCommand("bash -c 'gh pr create --title x'")).toBe(true)
  })

  test('does not match an unrelated gh subcommand', () => {
    expect(isPrCreateCommand('gh pr list')).toBe(false)
    expect(isPrCreateCommand('gh pr view 42 --json comments')).toBe(false)
    expect(isPrCreateCommand('gh api graphql -f query=x')).toBe(false)
  })

  test('does not match a command with no gh invocation at all', () => {
    expect(isPrCreateCommand('git commit -m "pr create"')).toBe(false)
  })

  test('does not match a gh subcommand that merely mentions pr in an argument', () => {
    expect(isPrCreateCommand('gh issue create --title "pr"')).toBe(false)
    expect(isPrCreateCommand('gh release create v1 --notes "closes the pr"')).toBe(false)
  })

  test('does not match a gh api call that names this hook file', () => {
    expect(
      isPrCreateCommand(
        'gh api repos/o/r/pulls/1/comments --jq \'.[] | select(.path=="pr-create-matcher.ts")\'',
      ),
    ).toBe(false)
  })
})
