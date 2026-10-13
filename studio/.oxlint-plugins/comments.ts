/**
 * Comment-policy rules from CLAUDE.md: English only, docstrings on exported functions,
 * and no shortcut disclosed in prose without a tracked issue behind it.
 */

import { type ESTree, defineRule } from '@oxlint/plugins'

// No `dos`, `das` or `para`: they collide with "DoS", "DOS" and "para-virtualized" in English
// technical prose, and a false positive here blocks a commit over a correct comment.
const PORTUGUESE_WORDS =
  /\b(?:não|então|porque|também|apenas|isso|esse|essa|agora|precisa|deve|está|são|seja|pode|mas|uma|pelo|pela|aqui|sempre|nunca|quando|onde)\b/i
const PORTUGUESE_DIACRITICS = /[ãõçáéíóúâêôàü]/i
// `temporary` only counts next to a noun that makes it a confession. Bare, it is an ordinary
// noun phrase in `temporary directory`, `temporary file`, `temporary buffer`, and blocking a
// commit over one of those is the same failure PORTUGUESE_WORDS avoids by excluding `dos` and
// `para`. Negation is not the distinction: `this is not a hack` fires just as much, and that
// term stays. Rationale tracked in #5.
const WORKAROUND_TERMS =
  /\b(?:todo|fixme|hack|xxx|workaround|for now|gambiarra|temporary\s+(?:fix|workaround|solution|patch|hack|shim|measure|implementation|approach))\b/i
const TRACKED_REFERENCE = /#\d+|https?:\/\//
const DOCSTRING_START = '*'

type CommentBlock = { text: string; parts: ESTree.Comment[] }

/**
 * Merges runs of adjacent `//` comments into one block, so a rule reading comment prose sees
 * what the author wrote rather than one AST node per line. Block comments never merge.
 * @param comments - every comment in the file, in source order
 * @returns one entry per block, carrying the joined text and the comments that formed it
 */
function consecutiveLineCommentBlocks(comments: readonly ESTree.Comment[]): CommentBlock[] {
  const blocks: CommentBlock[] = []
  let previous: ESTree.Comment | null = null

  for (const comment of comments) {
    const adjacent =
      previous !== null &&
      previous.type === 'Line' &&
      comment.type === 'Line' &&
      comment.loc.start.line === previous.loc.end.line + 1
    if (adjacent) {
      const block = blocks.at(-1)
      if (block !== undefined) {
        block.text += `\n${comment.value}`
        block.parts.push(comment)
      }
    } else {
      blocks.push({ text: comment.value, parts: [comment] })
    }
    previous = comment
  }
  return blocks
}

/** True when the initializer is a function, which is what makes an export worth documenting. */
function isFunctionValue(node: ESTree.Expression | null | undefined): boolean {
  return node?.type === 'ArrowFunctionExpression' || node?.type === 'FunctionExpression'
}

export const commentMustBeEnglish = defineRule({
  meta: {
    messages: {
      english: 'Write comments in English, even in a file that already has non-English ones.',
    },
  },
  create(context) {
    return {
      Program() {
        for (const comment of context.sourceCode.getAllComments()) {
          if (PORTUGUESE_WORDS.test(comment.value) || PORTUGUESE_DIACRITICS.test(comment.value)) {
            context.report({ loc: context.sourceCode.getLoc(comment), messageId: 'english' })
          }
        }
      },
    }
  },
})

export const workaroundNeedsIssueLink = defineRule({
  meta: {
    messages: {
      link: 'A workaround comment without an issue number or upstream link is a confession, not documentation.',
    },
  },
  create(context) {
    return {
      Program() {
        // Grouped, not per node: a run of `//` lines reads as one comment but parses as N,
        // so checking each in isolation flags `// TODO: refactor` for missing the `// see #12`
        // sitting right under it, with no way to satisfy the rule short of joining the lines.
        for (const block of consecutiveLineCommentBlocks(context.sourceCode.getAllComments())) {
          if (!WORKAROUND_TERMS.test(block.text) || TRACKED_REFERENCE.test(block.text)) continue
          // Point at the line that admits the shortcut, falling back to whatever opened the
          // block. A multi-word term can match across the newline this rule joins `//` lines
          // with while matching no single line, and without the fallback that block reports
          // nowhere: the confession written the way people actually write it, split over two
          // lines, would be the one shape the rule let through.
          const offender =
            block.parts.find((part) => WORKAROUND_TERMS.test(part.value)) ?? block.parts.at(0)
          if (offender !== undefined) {
            context.report({ loc: context.sourceCode.getLoc(offender), messageId: 'link' })
          }
        }
      },
    }
  },
})

/**
 * Requires a TSDoc block on every exported function, whether it is exported inline
 * (`export function foo`) or in a separate list (`function foo` plus `export { foo }`).
 * The separate form is why this collects candidates first and decides on `Program:exit`:
 * the export can appear before or after the declaration it names.
 */
export const requireTsdocOnExports = defineRule({
  meta: {
    messages: {
      document: 'Document this exported function with a TSDoc block describing what it does.',
    },
  },
  create(context) {
    const exportedNames = new Set<string>()
    const candidates = new Map<string, ESTree.Node>()

    const isDocumented = (node: ESTree.Node) =>
      context.sourceCode
        .getCommentsBefore(node)
        .some((comment) => comment.type === 'Block' && comment.value.startsWith(DOCSTRING_START))

    return {
      ExportNamedDeclaration(node) {
        for (const specifier of node.specifiers) {
          if (specifier.local.type === 'Identifier') exportedNames.add(specifier.local.name)
        }
        const { declaration } = node
        if (declaration === null) return

        // Inline export: the docblock sits before the `export` keyword, not the declaration.
        if (declaration.type === 'FunctionDeclaration' && declaration.id !== null) {
          if (!isDocumented(node)) context.report({ node, messageId: 'document' })
          return
        }
        if (declaration.type !== 'VariableDeclaration') return
        if (declaration.declarations.some((declarator) => isFunctionValue(declarator.init))) {
          if (!isDocumented(node)) context.report({ node, messageId: 'document' })
        }
      },
      // `export default function foo` and `function foo` plus `export default foo`. Rare here,
      // since import/no-default-export is an error outside config files, but a rule that only
      // half-covers exports is worse than one that says so.
      ExportDefaultDeclaration(node) {
        const { declaration } = node
        if (declaration.type === 'Identifier') {
          exportedNames.add(declaration.name)
          return
        }
        const exportsFunction =
          declaration.type === 'FunctionDeclaration' ||
          declaration.type === 'ArrowFunctionExpression' ||
          declaration.type === 'FunctionExpression'
        if (exportsFunction && !isDocumented(node)) {
          context.report({ node, messageId: 'document' })
        }
      },
      // Module scope only. `export { foo }` can only name a module-scope binding, so a nested
      // function sharing the name is a different symbol. Collecting it too would let the
      // later visit overwrite the real declaration, reporting the wrong node or going quiet
      // on a genuine violation, depending on which of the two carries the docblock.
      FunctionDeclaration(node) {
        if (node.id !== null && node.parent.type === 'Program') {
          candidates.set(node.id.name, node)
        }
      },
      // The declaration, not the declarator: getCommentsBefore only looks back to the previous
      // token, which for a declarator is the `const` keyword. Asking it about the declarator
      // can never see the docblock above the statement, so every documented arrow function
      // re-exported through `export { }` came back undocumented.
      VariableDeclarator(node) {
        const declaration = node.parent
        if (declaration.parent?.type !== 'Program') return
        if (node.id.type === 'Identifier' && isFunctionValue(node.init)) {
          candidates.set(node.id.name, declaration)
        }
      },
      'Program:exit'() {
        for (const [name, node] of candidates) {
          if (!exportedNames.has(name)) continue
          if (!isDocumented(node)) context.report({ node, messageId: 'document' })
        }
      },
    }
  },
})
