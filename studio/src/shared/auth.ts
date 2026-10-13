import { z } from 'zod'

const MAX_PASSWORD_LENGTH = 200

export const loginRequestSchema = z.object({ password: z.string().min(1).max(MAX_PASSWORD_LENGTH) })
export type LoginRequest = z.infer<typeof loginRequestSchema>

/** Whether this browser may use the API, and how it got there. */
export const sessionSchema = z.object({
  authenticated: z.boolean(),
  via: z.enum(['open', 'tailnet', 'password']).nullable(),
})
export type Session = z.infer<typeof sessionSchema>
