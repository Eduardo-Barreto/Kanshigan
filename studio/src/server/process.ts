/** Options for a child process whose output is consumed line by line. */
export type ProcessOptions = {
  cwd: string
  env?: Record<string, string>
  onLine: (line: string) => void
  signal: AbortSignal
}

// oxlint-disable-next-line no-control-regex -- ESC is exactly what an ANSI colour code starts with
const ANSI_ESCAPE = /\u001B\[[0-9;]*[A-Za-z]/g

async function pump(stream: ReadableStream<Uint8Array>, onLine: (line: string) => void) {
  const decoder = new TextDecoder()
  let buffered = ''
  for await (const chunk of stream) {
    buffered += decoder.decode(chunk, { stream: true })
    const lines = buffered.split(/[\r\n]+/)
    buffered = lines.pop() ?? ''
    lines
      .map((line) => line.replaceAll(ANSI_ESCAPE, ''))
      .filter((line) => line.trim() !== '')
      .forEach(onLine)
  }
  if (buffered.trim() !== '') onLine(buffered.replaceAll(ANSI_ESCAPE, ''))
}

/**
 * Runs a command to completion, feeding each line of stdout and stderr to `onLine`.
 * The command is an argument array, never a shell string, so no value is ever interpreted
 * by a shell. Rejects on a non-zero exit or when the signal aborts.
 */
export async function runProcess(command: string[], options: ProcessOptions): Promise<void> {
  const isCancelled = () => options.signal.aborted
  if (isCancelled()) throw new Error('cancelado')
  const child = Bun.spawn(command, {
    cwd: options.cwd,
    env: { ...process.env, ...options.env },
    stdout: 'pipe',
    stderr: 'pipe',
  })
  const abort = () => child.kill()
  options.signal.addEventListener('abort', abort, { once: true })
  try {
    await Promise.all([pump(child.stdout, options.onLine), pump(child.stderr, options.onLine)])
    const exitCode = await child.exited
    if (isCancelled()) throw new Error('cancelado')
    if (exitCode !== 0) throw new Error(`${command[0]} terminou com código ${exitCode}`)
  } finally {
    options.signal.removeEventListener('abort', abort)
  }
}
