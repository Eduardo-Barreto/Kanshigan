/**
 * The parent walk both `data-fetching` and `tests` need, shared so the two can't drift.
 * They previously carried the same loop with different depth caps, 30 and 20.
 */

import type { ESTree } from '@oxlint/plugins'

// A parent chain terminates at Program, so this is a bound against a malformed chain, not a
// statement about how deep real code nests. Pick generously, because the two consumers fail
// in opposite directions when it trips: a rule that reports on finding an ancestor goes quiet
// (no-fetch-in-effect misses the useEffect), and one that reports on finding none goes loud
// (require-top-level-describe flags a test that is properly nested).
const ANCESTOR_LIMIT = 30

/**
 * Walks up from `node` looking for an enclosing node that `matches` accepts.
 * @param node - where to start; its own parent is the first candidate, `node` itself is not tested
 * @param matches - predicate identifying the enclosing construct
 * @returns true when one is found within the depth bound
 */
export function hasAncestorMatching(
  node: ESTree.Node,
  matches: (candidate: ESTree.Node) => boolean,
): boolean {
  let current: ESTree.Node | null | undefined = 'parent' in node ? node.parent : undefined
  for (let depth = 0; depth < ANCESTOR_LIMIT; depth += 1) {
    if (current === null || current === undefined) return false
    if (matches(current)) return true
    current = 'parent' in current ? current.parent : undefined
  }
  return false
}
