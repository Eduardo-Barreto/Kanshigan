#!/usr/bin/env bun
// Lints only the file that was just edited and feeds the diagnostics straight back as
// context, so a violation gets fixed while the file is still fresh instead of surfacing
// in review much later. This hook informs, it never blocks: every failure path exits 0,
// because `bun run doctor` on the Stop hook is what actually gates the work.

import { spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { join } from 'node:path'

import { isRecord, readHookPayload } from './hook-input.ts'

const LINTABLE_FILE = /\.[jt]sx?$/
// Comfortably under this hook's own 30s timeout in settings.json, so the hook always gets
// to write its output instead of being killed mid-report.
const OXLINT_TIMEOUT_MS = 20_000
const MAX_DIAGNOSTICS = 20
const MAX_FAILURE_CHARS = 400
const LOCAL_OXLINT = join('node_modules', '.bin', 'oxlint')

type Diagnostic = { message: string; code: string; line: number; column: number }

function editedFilePath(): string | null {
  const payload = readHookPayload()
  if (payload === null || !isRecord(payload.tool_input)) return null
  const filePath = payload.tool_input.file_path
  return typeof filePath === 'string' ? filePath : null
}

function report(context: string): void {
  console.log(
    JSON.stringify({
      hookSpecificOutput: { hookEventName: 'PostToolUse', additionalContext: context },
    }),
  )
}

/**
 * Reads oxlint's `-f json` report.
 * @returns the diagnostics, or null when the output isn't a report at all, which is how a
 * config error or a crash arrives and is not the same thing as a clean file
 */
function parseDiagnostics(stdout: string): Diagnostic[] | null {
  let parsed: unknown
  try {
    parsed = JSON.parse(stdout)
  } catch {
    return null
  }
  if (!isRecord(parsed) || !Array.isArray(parsed.diagnostics)) return null

  return parsed.diagnostics.flatMap((entry: unknown) => {
    if (!isRecord(entry)) return []
    const labels: unknown[] = Array.isArray(entry.labels) ? entry.labels : []
    const [label] = labels
    const span = isRecord(label) && isRecord(label.span) ? label.span : {}
    return [
      {
        message: String(entry.message),
        code: String(entry.code),
        line: Number(span.line),
        column: Number(span.column),
      },
    ]
  })
}

const filePath = editedFilePath()
if (filePath === null || !LINTABLE_FILE.test(filePath)) process.exit(0)

const projectDir = process.env.CLAUDE_PROJECT_DIR
if (projectDir === undefined) process.exit(0)
process.chdir(projectDir)

// The installed binary, not `bunx oxlint`: bunx adds a resolution step that can fail or pick
// a different copy, and a hook that reports nothing when its linter never ran is
// indistinguishable from a clean file.
const [command, args] = existsSync(LOCAL_OXLINT)
  ? [LOCAL_OXLINT, ['-f', 'json', filePath]]
  : ['bunx', ['oxlint', '-f', 'json', filePath]]

const oxlint = spawnSync(command, args, { encoding: 'utf8', timeout: OXLINT_TIMEOUT_MS })

// Handled before reading the streams, because a process that never started is the one case
// where they come back null rather than empty, and because silence here would read as
// "this file is clean", which is the one thing it does not mean.
if (oxlint.error !== undefined) {
  report(`oxlint could not run on ${filePath}. This is a tooling failure: ${String(oxlint.error)}`)
  process.exit(0)
}

const diagnostics = parseDiagnostics(oxlint.stdout)
if (diagnostics === null) {
  const output = `${oxlint.stdout}${oxlint.stderr}`.trim().slice(0, MAX_FAILURE_CHARS)
  report(
    `oxlint could not lint ${filePath}. This is a tooling failure, not a clean file. Command: ${command}, exit ${String(oxlint.status)}.\n${output}`,
  )
  process.exit(0)
}

if (diagnostics.length > 0) {
  const lines = diagnostics
    .slice(0, MAX_DIAGNOSTICS)
    .map((entry) => `${filePath}:${entry.line}:${entry.column}: ${entry.code}: ${entry.message}`)
  report(`oxlint on ${filePath}:\n${lines.join('\n')}`)
}

process.exit(0)
