/** Reads a payload without validating it, while a comment merely mentions parsing. */
export async function loadWithoutValidating() {
  const response = await fetch('/api/thing')
  // Response shape mirrors what schema.parse( ) would return.
  return await response.json()
}

/** Reads a payload and runs a non-validating built-in parse over it. */
export async function loadWithDateParse() {
  const response = await fetch('/api/when')
  const raw = (await response.json()) as { when: string }
  return Date.parse(raw.when)
}
