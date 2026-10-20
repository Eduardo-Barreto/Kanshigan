/**
 * This project's coding conventions as real lint rules, so a diff is checked by an
 * AST instead of by an LLM reading CLAUDE.md. Registered via `jsPlugins` in .oxlintrc.json.
 */

import { definePlugin } from '@oxlint/plugins'

import {
  commentMustBeEnglish,
  requireTsdocOnExports,
  workaroundNeedsIssueLink,
} from './comments.ts'
import {
  formRequiresZodResolver,
  noFetchInEffect,
  noManualLoadingBranch,
  parseBeforeUse,
  queryKeyFromFactory,
} from './data-fetching.ts'
import { noManagerHelperSuffix, noStaticConstantInComponent, noVagueIdentifier } from './naming.ts'
import { noTestIdQuery, requireTopLevelDescribe, testTitleNoShould } from './tests.ts'

export default definePlugin({
  meta: { name: 'project' },
  rules: {
    'no-fetch-in-effect': noFetchInEffect,
    'query-key-from-factory': queryKeyFromFactory,
    'form-requires-zod-resolver': formRequiresZodResolver,
    'parse-before-use': parseBeforeUse,
    'no-manual-loading-branch': noManualLoadingBranch,
    'comment-must-be-english': commentMustBeEnglish,
    'workaround-needs-issue-link': workaroundNeedsIssueLink,
    'require-tsdoc-on-exports': requireTsdocOnExports,
    'no-manager-helper-suffix': noManagerHelperSuffix,
    'no-vague-identifier': noVagueIdentifier,
    'no-static-constant-in-component': noStaticConstantInComponent,
    'no-test-id-query': noTestIdQuery,
    'test-title-no-should': testTitleNoShould,
    'require-top-level-describe': requireTopLevelDescribe,
  },
})
