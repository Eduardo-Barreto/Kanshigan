/**
 * Testing-convention rules from CLAUDE.md. These exist as project rules because oxlint's
 * vitest plugin only recognises tests that import from `vitest`, and this repo runs `bun:test`.
 */

import { type ESTree, defineRule } from '@oxlint/plugins'

import { hasAncestorMatching } from './ancestors.ts'

const TEST_ID_QUERY = /^(?:get|query|find)(?:All)?ByTestId$/
const TEST_CALLERS = new Set(['test', 'it'])
const DESCRIBE_CALLER = 'describe'
const TABLE_BUILDERS = new Set(['each', 'for'])
// Allow-list, not a deny-list: `test.describe` is a describe and `test.beforeEach` is a hook,
// and treating either as a test case makes require-top-level-describe fire on valid specs.
const TEST_MODIFIERS = new Set([
  'only',
  'skip',
  'todo',
  'failing',
  'fails',
  'each',
  'for',
  'concurrent',
  'serial',
  'skipIf',
  'runIf',
  'todoIf',
])
const SHOULD_TITLE = /^should\b/i

/**
 * Resolves what a call ultimately calls, through modifiers and table forms.
 * @returns 'test', 'it' or 'describe' for `test(...)`, `it.only(...)`, `it.each([...])(...)`,
 * `describe.each([...])(...)` and Playwright's `test.describe(...)`; null for a hook like
 * `test.beforeEach` and for everything that isn't a test construct
 */
function resolveCaller(callee: ESTree.Expression): string | null {
  if (callee.type === 'Identifier') return callee.name
  // it.each([...])('title', fn): the callee is itself a call.
  if (callee.type === 'CallExpression') return resolveCaller(callee.callee)
  if (callee.type !== 'MemberExpression' || callee.computed) return null
  // Playwright nests the describe under the test object, so the property wins here.
  if (callee.property.name === DESCRIBE_CALLER) return DESCRIBE_CALLER
  if (!TEST_MODIFIERS.has(callee.property.name)) return null
  return resolveCaller(callee.object)
}

/**
 * True for the builder half of a table form: `it.each(table)`, not the `it.each(table)(...)`
 * that runs. Both resolve to the same caller, so a rule that counts either as the construct
 * itself sees one test twice, or treats a builder as the block it is an argument to.
 */
function isTableBuilder(node: ESTree.CallExpression): boolean {
  return (
    node.callee.type === 'MemberExpression' &&
    !node.callee.computed &&
    TABLE_BUILDERS.has(node.callee.property.name)
  )
}

function isTestCall(node: ESTree.CallExpression): boolean {
  if (isTableBuilder(node)) return false
  const caller = resolveCaller(node.callee)
  return caller !== null && TEST_CALLERS.has(caller)
}

function isDescribeCall(node: ESTree.Node): boolean {
  if (node.type !== 'CallExpression' || isTableBuilder(node)) return false
  return resolveCaller(node.callee) === DESCRIBE_CALLER
}

export const noTestIdQuery = defineRule({
  meta: {
    messages: {
      useRole:
        'Query by role or label, not by test-id. A test-id tests the DOM, not the user-visible behaviour.',
    },
  },
  create(context) {
    return {
      MemberExpression(node) {
        if (node.computed) return
        if (TEST_ID_QUERY.test(node.property.name)) {
          context.report({ node: node.property, messageId: 'useRole' })
        }
      },
    }
  },
})

export const testTitleNoShould = defineRule({
  meta: {
    messages: {
      thirdPerson: 'Name the behaviour with a third-person verb, not "should".',
    },
  },
  create(context) {
    return {
      CallExpression(node) {
        if (!isTestCall(node)) return
        const [title] = node.arguments
        if (title?.type === 'Literal' && typeof title.value === 'string') {
          if (SHOULD_TITLE.test(title.value)) {
            context.report({ node: title, messageId: 'thirdPerson' })
          }
        }
      },
    }
  },
})

export const requireTopLevelDescribe = defineRule({
  meta: {
    messages: {
      wrap: 'Wrap this test in a describe block that names the behaviour it segments.',
    },
  },
  create(context) {
    return {
      CallExpression(node) {
        if (!isTestCall(node)) return
        if (hasAncestorMatching(node, isDescribeCall)) return
        context.report({ node, messageId: 'wrap' })
      },
    }
  },
})
