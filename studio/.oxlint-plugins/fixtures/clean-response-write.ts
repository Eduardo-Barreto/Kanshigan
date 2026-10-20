type ResponseWriter = { json: (payload: unknown) => Response }

/**
 * Writes a JSON response. `json` here takes the payload as an argument, so it serializes
 * rather than reads: there is no untyped boundary to validate.
 */
export function writeGreeting(context: ResponseWriter) {
  return context.json({ greeting: 'hello' })
}
