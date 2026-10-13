type LogFields = Record<string, string | number | boolean | null | undefined>

const LEVELS = ['info', 'warn', 'error'] as const
type LogLevel = (typeof LEVELS)[number]

const isSilenced = process.env.NODE_ENV === 'test'

function write(level: LogLevel, event: string, fields: LogFields) {
  if (isSilenced) return
  const line = JSON.stringify({ time: new Date().toISOString(), level, event, ...fields })
  process.stdout.write(`${line}\n`)
}

/** Structured JSON-lines logger: one event name plus flat fields, greppable and machine-readable. */
export const log = {
  info: (event: string, fields: LogFields = {}) => write('info', event, fields),
  warn: (event: string, fields: LogFields = {}) => write('warn', event, fields),
  error: (event: string, fields: LogFields = {}) => write('error', event, fields),
}

/** Runs `operation`, logging its duration and outcome under `event`. Rethrows on failure. */
export async function withTiming<Result>(
  event: string,
  fields: LogFields,
  operation: () => Promise<Result>,
): Promise<Result> {
  const startedAt = performance.now()
  try {
    const result = await operation()
    log.info(event, { ...fields, durationMs: Math.round(performance.now() - startedAt) })
    return result
  } catch (error) {
    log.error(event, {
      ...fields,
      durationMs: Math.round(performance.now() - startedAt),
      error: error instanceof Error ? error.message : String(error),
    })
    throw error
  }
}
