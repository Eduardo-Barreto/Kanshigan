import { describe, expect, test } from 'bun:test'
import { spawnSync } from 'node:child_process'

import { z } from 'zod'

const FIXTURES_DIR = '.oxlint-plugins/fixtures'
const FIXTURES_CONFIG = `${FIXTURES_DIR}/oxlintrc.json`
const RULE_CODE = /^project\((?<rule>[a-z-]+)\)$/
const OXLINT_TIMEOUT_MS = 30_000

const oxlintReport = z.looseObject({
  diagnostics: z.array(z.looseObject({ code: z.string(), filename: z.string() })),
})

type Finding = { rule: string; file: string }

/**
 * Lints the fixture directory with the project's real config and returns only this plugin's
 * findings. Fixtures need their own config because the production one ignores them, which is
 * what keeps `bun run doctor` from failing on deliberate violations.
 */
function lintFixtures(): Finding[] {
  const oxlint = spawnSync('bunx', ['oxlint', '-c', FIXTURES_CONFIG, '-f', 'json', FIXTURES_DIR], {
    encoding: 'utf8',
    timeout: OXLINT_TIMEOUT_MS,
  })
  // Checked before parsing: a killed or never-started process returns null streams, and
  // JSON.parse(null) throws something that says nothing about what actually went wrong.
  if (oxlint.error !== undefined) {
    throw new Error(`oxlint did not run: ${String(oxlint.error)}`)
  }
  // Not the timeout case, which sets `error` and is caught above: this is oxlint or bunx
  // writing non-JSON on a config error or a panic. A bare SyntaxError from JSON.parse beats
  // the descriptive throw below to the exit, hiding the output worth reading in CI.
  let parsed: unknown
  try {
    parsed = JSON.parse(oxlint.stdout)
  } catch (error) {
    // The parser's own message carries the offending position, which a large stdout makes
    // expensive to find by eye.
    const reason = error instanceof Error ? error.message : String(error)
    throw new Error(
      `oxlint returned unparseable output (${reason}): ${oxlint.stdout}${oxlint.stderr}`,
      { cause: error },
    )
  }
  const report = oxlintReport.safeParse(parsed)
  if (!report.success) {
    throw new Error(`oxlint returned no usable diagnostics: ${oxlint.stdout}${oxlint.stderr}`)
  }

  return report.data.diagnostics.flatMap((diagnostic) => {
    const rule = RULE_CODE.exec(diagnostic.code)?.groups?.rule
    return rule === undefined ? [] : [{ rule, file: diagnostic.filename }]
  })
}

const findings = lintFixtures()
const rulesIn = (fixture: string) =>
  findings.filter((finding) => finding.file.includes(fixture)).map((finding) => finding.rule)
const countIn = (fixture: string, rule: string) =>
  rulesIn(fixture).filter((found) => found === rule).length

describe('data-fetching rules', () => {
  const flagged = rulesIn('violates-data-fetching')

  test('flags fetching inside useEffect', () => {
    expect(flagged).toContain('no-fetch-in-effect')
  })

  test('flags a queryKey built from a bare string', () => {
    expect(flagged).toContain('query-key-from-factory')
  })

  test('flags useForm without a resolver', () => {
    expect(flagged).toContain('form-requires-zod-resolver')
  })

  test('flags a payload read without a Zod parse', () => {
    expect(flagged).toContain('parse-before-use')
  })

  test('reaches a method as well as a function, reporting each once', () => {
    expect(countIn('violates-data-fetching', 'parse-before-use')).toBe(2)
  })

  test('does not accept a comment mentioning parse as the validation itself', () => {
    expect(rulesIn('violates-parse-in-comment')).toContain('parse-before-use')
  })

  test('does not accept a non-validating built-in parse as validation', () => {
    expect(countIn('violates-parse-in-comment', 'parse-before-use')).toBe(2)
  })

  // Both receivers satisfied the old text match: `payloadCodec` because its name sat outside
  // the deny-list, `currentUrl` because the deny-list's word boundary never matched a name
  // that merely ended in Url.
  test('does not accept a parse on a receiver that is not a schema', () => {
    expect(countIn('violates-parse-receivers', 'parse-before-use')).toBe(2)
  })

  test('resolves a schema by its z initializer, not by a Schema suffix in its name', () => {
    expect(countIn('clean-local-schema', 'parse-before-use')).toBe(0)
  })

  test('reads through a cast and through a this-rooted field to the receiver behind them', () => {
    expect(countIn('clean-wrapped-schema', 'parse-before-use')).toBe(0)
  })

  test('does not treat a json call that writes a response as a payload read', () => {
    expect(countIn('clean-response-write', 'parse-before-use')).toBe(0)
  })

  test('flags a manual isLoading branch', () => {
    expect(flagged).toContain('no-manual-loading-branch')
  })
})

describe('comment rules', () => {
  const flagged = rulesIn('violates-comments')

  test('flags a comment written in Portuguese', () => {
    expect(flagged).toContain('comment-must-be-english')
  })

  test('flags a TODO with no issue reference', () => {
    expect(flagged).toContain('workaround-needs-issue-link')
  })

  test('accepts an issue reference on the line below the workaround', () => {
    expect(countIn('clean-split-workaround', 'workaround-needs-issue-link')).toBe(0)
  })

  // The pair, not either alone: `temporary` bare is `temporary directory`, and dropping it
  // outright would have lost `temporary fix`, the confession the rule exists for. See #5.
  test('leaves temporary as an ordinary noun phrase out of the workaround terms', () => {
    expect(countIn('clean-temporary-noun', 'workaround-needs-issue-link')).toBe(0)
  })

  test('still flags temporary next to the noun that makes it a confession', () => {
    expect(countIn('violates-temporary-fix', 'workaround-needs-issue-link')).toBe(1)
  })

  // The multi-word term matches across the newline joining two `//` lines while matching
  // neither line alone, so the block had nowhere to report and stayed silent.
  test('flags a confession split across two comment lines', () => {
    expect(countIn('violates-split-confession', 'workaround-needs-issue-link')).toBe(1)
  })

  test('flags an exported function with no TSDoc', () => {
    expect(flagged).toContain('require-tsdoc-on-exports')
  })

  test('reaches a function exported in a separate export list', () => {
    expect(countIn('violates-comments', 'require-tsdoc-on-exports')).toBe(3)
  })

  test('sees the docblock above a re-exported arrow function', () => {
    expect(countIn('clean-documented-arrow-export', 'require-tsdoc-on-exports')).toBe(0)
  })

  test('reaches a function reached only through a default export', () => {
    expect(rulesIn('violates-default-export')).toContain('require-tsdoc-on-exports')
  })

  test('resolves an exported name to its module-scope declaration, not a nested shadow', () => {
    expect(countIn('clean-shadowed-names', 'require-tsdoc-on-exports')).toBe(0)
  })
})

describe('naming rules', () => {
  const flagged = rulesIn('violates-naming')

  test('flags a Manager suffix', () => {
    expect(flagged).toContain('no-manager-helper-suffix')
  })

  test('flags a vague identifier', () => {
    expect(flagged).toContain('no-vague-identifier')
  })

  test('flags a static constant declared inside a component', () => {
    expect(flagged).toContain('no-static-constant-in-component')
  })

  test('reaches an arrow component as well as a declared one', () => {
    expect(countIn('violates-naming', 'no-static-constant-in-component')).toBe(2)
  })
})

describe('test rules', () => {
  const flagged = rulesIn('violates-tests')

  test('flags a query by test-id', () => {
    expect(flagged).toContain('no-test-id-query')
  })

  test('flags a title starting with should', () => {
    expect(flagged).toContain('test-title-no-should')
  })

  test('flags a test outside any describe block', () => {
    expect(flagged).toContain('require-top-level-describe')
  })

  test('reaches a title behind a table form like it.each', () => {
    expect(rulesIn('violates-tests-table')).toContain('test-title-no-should')
  })

  // Counts, not toContain: both of these pass either way as membership checks, and the two
  // table-form guards they pin were deletable green before these existed.
  test('counts a table form outside a describe once, not once per half', () => {
    expect(countIn('violates-table-outside-describe', 'require-top-level-describe')).toBe(1)
  })

  test('does not accept an uninvoked describe.each builder as the describe it would build', () => {
    expect(countIn('violates-uninvoked-builder', 'require-top-level-describe')).toBe(1)
  })

  test('leaves a playwright hook like test.beforeEach out of the describe requirement', () => {
    expect(rulesIn('clean-playwright')).toEqual([])
  })

  test('recognises a table-form describe.each as a describe', () => {
    expect(rulesIn('clean-tests')).toEqual([])
  })
})

describe('conforming code', () => {
  test('reports nothing on a file that follows every convention', () => {
    expect(rulesIn('clean')).toEqual([])
  })
})
