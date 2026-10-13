const queues = new Map<string, Promise<unknown>>()

/**
 * Runs operations that share a key one at a time, in arrival order. Two saves to the same
 * clip read-modify-write the same sidecar file, so they must not interleave.
 */
export async function serialize<Result>(
  key: string,
  operation: () => Promise<Result>,
): Promise<Result> {
  const previous = queues.get(key) ?? Promise.resolve()
  const next = (async () => {
    await previous.catch(() => undefined)
    return operation()
  })()
  queues.set(key, next)
  try {
    return await next
  } finally {
    if (queues.get(key) === next) queues.delete(key)
  }
}
