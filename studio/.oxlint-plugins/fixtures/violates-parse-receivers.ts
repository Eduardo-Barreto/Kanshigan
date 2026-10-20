// The two receivers the old text match let through. Neither is a schema, and both used to
// satisfy the rule: one because its name sat outside a fixed deny-list, the other because the
// deny-list's word boundary never matched a name that merely ended in `Url`.

const payloadCodec = { parse: (input: unknown) => ({ id: String(input) }) }
const currentUrl = { parse: (input: unknown) => String(input) }

/** Reads a payload and hands it to a codec, which converts rather than validates. */
export async function loadThroughCodec(response: Response) {
  const raw: unknown = await response.json()
  return payloadCodec.parse(raw)
}

/** Reads a payload and hands it to a binding whose name ends in Url. */
export async function loadThroughUrl(response: Response) {
  const raw: unknown = await response.json()
  return currentUrl.parse(raw)
}
