import { z } from 'zod'

const memberSchema = z.object({ id: z.string() })

/** Validates through a cast, which sits between the receiver and the name behind it. */
export async function loadThroughCast(response: Response) {
  const raw: unknown = await response.json()
  return (memberSchema as z.ZodType<{ id: string }>).parse(raw)
}

/** Validates through a field, where the chain roots at `this` rather than at a binding. */
export class MemberLoader {
  private readonly schema = memberSchema

  /** Reads a payload and validates it with the schema held on this instance. */
  async load(response: Response) {
    const raw: unknown = await response.json()
    return this.schema.parse(raw)
  }
}
