import { readFileSync } from 'node:fs'

/** Narrows an unknown value to a plain object. */
export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** Parses the hook's JSON payload from stdin (fd 0), returning null on any malformed input. */
export function readHookPayload(): Record<string, unknown> | null {
  try {
    const payload: unknown = JSON.parse(readFileSync(0, 'utf8'))
    return isRecord(payload) ? payload : null
  } catch {
    return null
  }
}

/**
 * Reads CLAUDE_PROJECT_DIR and chdirs into it. Prints an error naming
 * `purpose` and exits with `exitCode` if it isn't set.
 */
export function requireProjectDir(purpose: string, exitCode: number): void {
  const projectDir = process.env.CLAUDE_PROJECT_DIR
  if (!projectDir) {
    console.error(`CLAUDE_PROJECT_DIR is not set; refusing to ${purpose}.`)
    process.exit(exitCode)
  }
  process.chdir(projectDir)
}

/**
 * Runs `main`, exiting 0 if it settles normally. A thrown error or a rejected promise prints
 * its message and exits 2 (fail closed) instead of letting an uncaught exception exit 1,
 * which Claude Code treats as non-blocking.
 *
 * Async so a hook can load a dependency from outside `.claude/hooks/` inside the guard. A
 * static import that fails to resolve throws during module load, before this guard exists,
 * and exit 1 would open the gate the hook is there to close.
 */
export async function runHookGuard(main: () => void | Promise<void>): Promise<never> {
  try {
    await main()
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error))
    process.exit(2)
  }
  process.exit(0)
}
