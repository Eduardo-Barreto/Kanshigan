// Pure PR-create detection, split out from require-review.ts so it's
// importable and testable without touching stdin or the findings directory.

/**
 * Detects a `gh pr create` invocation, or the equivalent raw REST route
 * (`gh api .../pulls` with a request that implies POST), anywhere in a raw
 * Bash command string, including inside quoted arguments. Shell quoting
 * doesn't change what a command actually executes, so stripping quotes
 * before matching would let a trivially requoted `gh "pr" "create"` execute
 * for real while reading as plain text with no match. The two tokens are
 * matched adjacently, allowing a quote between them, because `create` is a
 * subcommand of `pr` and nothing valid sits in between. Searching for the
 * words independently blocked commands that merely mention both somewhere
 * (`gh issue create --title "pr"`, a `gh api` call naming this very file).
 *
 * This is a deliberately bounded, best-effort check, not a sandbox: bash
 * offers unlimited ways to splice a command's own text apart while it still
 * executes identically (`gh pr cre""ate`, `gh "p"r create`, a backslash
 * mid-word, a wrapper script invoked as `bash /tmp/x.sh`). Two other classes
 * bypass by design, not by accident: a client that isn't `gh` at all (a raw
 * `curl -X POST` against the REST API) fails the very first `gh` check, and
 * a `gh` invocation shaped differently from the two forms this function
 * actually recognizes (`gh api graphql`'s `createPullRequest` mutation is a
 * real `gh` call, but matches neither `pr create` nor the `/pulls` route)
 * passes straight through. Fully closing either class needs enforcement at
 * exec time (for example, a `gh` wrapper earlier in PATH checking against
 * the parsed argv the process actually receives), not a better regex. The
 * primary defense against a diff engineering its way past this check is
 * each reviewer's own "diff content is data, not instructions" boundary;
 * this matcher is a mechanical backstop for the common case, not the only
 * line of defense.
 * @param rawCommand - the exact command text a Bash tool call is about to run
 */
export function isPrCreateCommand(rawCommand: string): boolean {
  if (!/\bgh\b/.test(rawCommand)) return false
  const prCreate = /\bpr\b["']?\s+["']?create\b/.test(rawCommand)
  // `gh api` defaults to GET, but an explicit -X/--method POST isn't the only
  // way to get one: passing -f/-F/--field/--raw-field/--input supplies a
  // request body, which gh switches to POST for automatically.
  const impliesPost =
    /(-X\s*POST|--method[= ]?POST)/i.test(rawCommand) ||
    /(^|\s)(-f|-F|--field|--raw-field|--input)(=|\s|$)/.test(rawCommand)
  // The collection route (creates a PR) only, not a sub-resource under a
  // specific PR number (`/pulls/123/comments`, a reply or review action).
  // The route argument is often quoted (`"repos/.../pulls"`), so a closing
  // quote counts as a valid terminator alongside whitespace, `?`, and the
  // end of the string.
  const apiPullsPost =
    /\bapi\b/.test(rawCommand) && /\/pulls(?:[\s?"']|$)/.test(rawCommand) && impliesPost
  return prCreate || apiPullsPost
}
