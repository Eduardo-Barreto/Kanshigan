/**
 * Naming and placement rules from CLAUDE.md: descriptive names, no habitual suffixes,
 * and no static constant trapped inside a component body.
 */

import { type ESTree, defineRule } from '@oxlint/plugins'

const HABITUAL_SUFFIX = /(?:Manager|Helper|Service|Utils?)$/
const VAGUE_NAMES = new Set([
  'data',
  'item',
  'list',
  'info',
  'mgr',
  'temp',
  'tmp',
  'val',
  'obj',
  'arr',
  'thing',
  'stuff',
  'res',
  'req',
])
const COMPONENT_NAME = /^[A-Z]/

export const noManagerHelperSuffix = defineRule({
  meta: {
    messages: {
      rename:
        'Drop the Manager/Helper/Service/Util suffix and name what it does. See CLAUDE.md naming.',
    },
  },
  create(context) {
    const check = (node: ESTree.Node, name: string) => {
      if (HABITUAL_SUFFIX.test(name)) {
        context.report({ node, messageId: 'rename' })
      }
    }

    return {
      FunctionDeclaration(node) {
        if (node.id !== null) check(node.id, node.id.name)
      },
      VariableDeclarator(node) {
        if (node.id.type === 'Identifier') check(node.id, node.id.name)
      },
    }
  },
})

/**
 * Flags placeholder names, including a shorthand destructure that keeps a library's
 * generic key. `const { data } = useQuery()` is flagged; `const { data: members }` is not.
 */
export const noVagueIdentifier = defineRule({
  meta: {
    messages: {
      describe: 'Name this for what it holds. Vague names like data, item or info hide intent.',
    },
  },
  create(context) {
    return {
      VariableDeclarator(node) {
        if (node.id.type === 'Identifier' && VAGUE_NAMES.has(node.id.name)) {
          context.report({ node: node.id, messageId: 'describe' })
          return
        }
        if (node.id.type !== 'ObjectPattern') return
        for (const property of node.id.properties) {
          if (property.type !== 'Property' || !property.shorthand) continue
          if (property.key.type === 'Identifier' && VAGUE_NAMES.has(property.key.name)) {
            context.report({ node: property, messageId: 'describe' })
          }
        }
      },
    }
  },
})

/** True when the expression is a literal, or an array/object built only from literals. */
function isStaticValue(node: ESTree.Expression | null): boolean {
  if (node === null) return false
  if (node.type === 'Literal') return true
  if (node.type === 'ArrayExpression') {
    return node.elements.every(
      (element) => element !== null && element.type !== 'SpreadElement' && isStaticValue(element),
    )
  }
  if (node.type === 'ObjectExpression') {
    return node.properties.every(
      (property) => property.type === 'Property' && isStaticValue(property.value),
    )
  }
  return false
}

export const noStaticConstantInComponent = defineRule({
  meta: {
    messages: {
      hoist:
        'Move this constant to module scope. Declaring it in the component body rebuilds it every render.',
    },
  },
  create(context) {
    const reportStaticDeclarations = (body: ESTree.FunctionBody | null) => {
      for (const statement of body?.body ?? []) {
        // `const` only: a `let` can be reassigned during the render, and hoisting it to module
        // scope would share that mutation across every instance of the component.
        if (statement.type !== 'VariableDeclaration' || statement.kind !== 'const') continue
        for (const declarator of statement.declarations) {
          if (isStaticValue(declarator.init ?? null)) {
            context.report({ node: declarator, messageId: 'hoist' })
          }
        }
      }
    }

    return {
      FunctionDeclaration(node) {
        if (node.id === null || !COMPONENT_NAME.test(node.id.name)) return
        reportStaticDeclarations(node.body ?? null)
      },
      // `const Panel = () => {...}`: an arrow component is a component all the same.
      VariableDeclarator(node) {
        if (node.id.type !== 'Identifier' || !COMPONENT_NAME.test(node.id.name)) return
        const initializer = node.init
        if (
          initializer?.type !== 'ArrowFunctionExpression' &&
          initializer?.type !== 'FunctionExpression'
        ) {
          return
        }
        if (initializer.body?.type !== 'BlockStatement') return
        reportStaticDeclarations(initializer.body)
      },
    }
  },
})
