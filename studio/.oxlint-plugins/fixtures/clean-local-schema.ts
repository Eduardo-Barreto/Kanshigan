import { z } from 'zod'

/**
 * Reads a payload and validates it with a schema declared below. The order is deliberate: the
 * rule resolves receivers at Program:exit precisely so a schema can appear after the function
 * that parses with it.
 */
export async function loadValidated(response: Response) {
  const raw: unknown = await response.json()
  return editablePayload.parse(raw)
}

// The name does not end in Schema, which is how this repo actually names them
// (`editablePackageJson` in scripts/init.ts). Resolved by its initializer, not by convention.
const editablePayload = z.object({ id: z.string() })
