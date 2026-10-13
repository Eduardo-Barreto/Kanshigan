/**
 * Rules for this project's data-fetching and form conventions, the ones CLAUDE.md
 * states and no off-the-shelf lint rule checks.
 */

import { type ESTree, defineRule } from '@oxlint/plugins'

import { hasAncestorMatching } from './ancestors.ts'

const RESPONSE_READ_METHOD = 'json'
const PARSE_METHODS = new Set(['parse', 'safeParse', 'parseAsync', 'safeParseAsync'])
const ZOD_NAMESPACE = 'z'
// Fallback for a schema imported from another module, where the initializer isn't in this file
// to resolve. A local `const x = z.object(...)` is recognised by its initializer instead.
const SCHEMA_NAME = /[Ss]chema$/
const RECEIVER_DEPTH_LIMIT = 20

/**
 * Strips type-only syntax wrapping a receiver, so a cast or a non-null assertion can't hide
 * the binding behind it.
 * @returns the wrapped expression, or null when `node` is not one of those wrappers
 */
function withoutTypeSyntax(node: ESTree.Node): ESTree.Node | null {
  if (node.type === 'TSAsExpression') return node.expression
  if (node.type === 'TSSatisfiesExpression') return node.expression
  if (node.type === 'TSNonNullExpression') return node.expression
  if (node.type === 'TSInstantiationExpression') return node.expression
  return null
}

/**
 * Leftmost identifier a member or call chain is rooted at: `z` for `z.object({}).parse`,
 * `JSON` for `JSON.parse`, `memberSchema` for `memberSchema.safeParse`.
 * @returns the name, or null when the chain is rooted at something that isn't an identifier
 */
function rootIdentifier(node: ESTree.Expression): string | null {
  let current: ESTree.Node = node
  for (let depth = 0; depth < RECEIVER_DEPTH_LIMIT; depth += 1) {
    if (current.type === 'Identifier') return current.name
    const unwrapped = withoutTypeSyntax(current)
    if (unwrapped !== null) {
      current = unwrapped
      continue
    }
    if (current.type === 'MemberExpression') {
      // `this.schema.parse(...)` roots at `this`, which names nothing. The property sitting
      // next to it is the binding a reader would call the schema.
      if (current.object.type === 'ThisExpression' && !current.computed) {
        return current.property.type === 'Identifier' ? current.property.name : null
      }
      current = current.object
      continue
    }
    if (current.type === 'CallExpression') {
      current = current.callee
      continue
    }
    return null
  }
  return null
}

/** True for a non-computed `.<name>(...)` call, the only shape either check cares about. */
function methodCall(
  node: ESTree.CallExpression,
): { receiver: ESTree.Expression; method: string } | null {
  const { callee } = node
  if (callee.type !== 'MemberExpression' || callee.computed) return null
  if (callee.property.type !== 'Identifier') return null
  return { receiver: callee.object, method: callee.property.name }
}

/** True for a `useEffect(...)` call, the ancestor no-fetch-in-effect is looking for. */
function isUseEffectCall(node: ESTree.Node): boolean {
  return (
    node.type === 'CallExpression' &&
    node.callee.type === 'Identifier' &&
    node.callee.name === 'useEffect'
  )
}

/** True for `fetch(...)`, `window.fetch(...)` and any `axios.get(...)`-style method. */
function isNetworkCall(callee: ESTree.Expression): boolean {
  if (callee.type === 'Identifier') return callee.name === 'fetch'
  if (callee.type !== 'MemberExpression' || callee.computed) return false
  if (callee.property.name === 'fetch') return true
  return callee.object.type === 'Identifier' && callee.object.name === 'axios'
}

export const noFetchInEffect = defineRule({
  meta: {
    messages: {
      useQuery: 'Fetch with TanStack Query, not inside useEffect. See CLAUDE.md.',
    },
  },
  create(context) {
    return {
      // Matched on the AST rather than on the effect's source text: a string literal that
      // merely contains "fetch(" is not a network call, and a comment about fetching is not
      // one either.
      CallExpression(node) {
        if (!isNetworkCall(node.callee)) return
        if (hasAncestorMatching(node, isUseEffectCall)) {
          context.report({ node, messageId: 'useQuery' })
        }
      },
    }
  },
})

export const queryKeyFromFactory = defineRule({
  meta: {
    messages: {
      useFactory:
        'Build queryKey from a key factory or enum, not a bare string. Magic cache keys drift silently.',
    },
  },
  create(context) {
    return {
      Property(node) {
        if (node.key.type !== 'Identifier' || node.key.name !== 'queryKey') return
        if (node.value.type !== 'ArrayExpression') return
        const [first] = node.value.elements
        if (first?.type === 'Literal' && typeof first.value === 'string') {
          context.report({ node: first, messageId: 'useFactory' })
        }
      },
    }
  },
})

export const formRequiresZodResolver = defineRule({
  meta: {
    messages: {
      addResolver: 'useForm needs a zodResolver: unvalidated form input is an untyped boundary.',
    },
  },
  create(context) {
    return {
      CallExpression(node) {
        if (node.callee.type !== 'Identifier' || node.callee.name !== 'useForm') return
        const [options] = node.arguments
        // A `resolver` key isn't enough: `resolver: undefined` and `resolver: yupResolver(x)`
        // both satisfy the shape while leaving the form outside the contract this rule names.
        const hasResolver =
          options?.type === 'ObjectExpression' &&
          options.properties.some(
            (property) =>
              property.type === 'Property' &&
              property.key.type === 'Identifier' &&
              property.key.name === 'resolver' &&
              property.value.type === 'CallExpression' &&
              property.value.callee.type === 'Identifier' &&
              property.value.callee.name === 'zodResolver',
          )
        if (!hasResolver) {
          context.report({ node, messageId: 'addResolver' })
        }
      },
    }
  },
})

/**
 * Flags a function that reads a fetch payload without validating it anywhere in the same
 * function. Scoped to the function, not the expression, so `const raw = await res.json()`
 * followed by `schema.parse(raw)` stays clean.
 *
 * Matched on the AST, not on source text. A text match could not tell `JSON.parse` from a
 * schema's without a lookbehind, and the lookbehind it used treated `myUrl.parse` as
 * validation while accepting any receiver outside a fixed deny-list. Reading the receiver off
 * the AST makes the question exact: is this chain rooted at `z`, at a binding this file
 * initialises from `z`, or at a name ending in Schema?
 *
 * Covers declarations, arrows and function expressions (object and class methods). When the
 * offender nests, only the innermost function is reported: blaming the enclosing one too
 * points at a whole function body for a problem that lives on one line inside it.
 */
export const parseBeforeUse = defineRule({
  meta: {
    messages: {
      validate:
        'Validate this payload with a Zod schema before using it. An unparsed boundary is an any in disguise.',
    },
  },
  create(context) {
    const functions: ESTree.Node[] = []
    const responseReads: ESTree.Node[] = []
    const parseCalls: { node: ESTree.Node; receiver: string | null }[] = []
    const localSchemas = new Set<string>()

    const collect = (node: ESTree.Node) => {
      functions.push(node)
    }

    const contains = (outer: ESTree.Node, inner: ESTree.Node) => {
      const [outerStart, outerEnd] = context.sourceCode.getRange(outer)
      const [innerStart, innerEnd] = context.sourceCode.getRange(inner)
      return outerStart <= innerStart && outerEnd >= innerEnd
    }

    const validates = (receiver: string | null) =>
      receiver !== null &&
      (receiver === ZOD_NAMESPACE || localSchemas.has(receiver) || SCHEMA_NAME.test(receiver))

    return {
      FunctionDeclaration: collect,
      ArrowFunctionExpression: collect,
      FunctionExpression: collect,
      // Collected, not resolved here: a schema can be declared below the function that parses
      // with it, so the set is only complete at Program:exit.
      VariableDeclarator(node) {
        if (node.id.type !== 'Identifier' || node.init === null) return
        if (rootIdentifier(node.init) === ZOD_NAMESPACE) localSchemas.add(node.id.name)
      },
      CallExpression(node) {
        const call = methodCall(node)
        if (call === null) return
        // `response.json()` reads a payload and takes no arguments. A call that passes one,
        // like a server framework's `context.json(payload)`, writes a response instead.
        if (call.method === RESPONSE_READ_METHOD && node.arguments.length === 0) {
          responseReads.push(node)
          return
        }
        if (PARSE_METHODS.has(call.method)) {
          parseCalls.push({ node, receiver: rootIdentifier(call.receiver) })
        }
      },
      'Program:exit'() {
        const offenders = functions.filter(
          (fn) =>
            responseReads.some((read) => contains(fn, read)) &&
            !parseCalls.some((parse) => validates(parse.receiver) && contains(fn, parse.node)),
        )
        for (const offender of offenders) {
          const nested = offenders.some((other) => other !== offender && contains(offender, other))
          if (!nested) {
            context.report({ node: offender, messageId: 'validate' })
          }
        }
      },
    }
  },
})

export const noManualLoadingBranch = defineRule({
  meta: {
    messages: {
      useSuspense:
        'Render loading state with <Suspense> and useSuspenseQuery instead of branching on isLoading.',
    },
  },
  create(context) {
    return {
      LogicalExpression(node) {
        if (node.operator !== '&&') return
        if (node.left.type === 'Identifier' && node.left.name === 'isLoading') {
          context.report({ node: node.left, messageId: 'useSuspense' })
        }
      },
      ConditionalExpression(node) {
        if (node.test.type === 'Identifier' && node.test.name === 'isLoading') {
          context.report({ node: node.test, messageId: 'useSuspense' })
        }
      },
    }
  },
})
