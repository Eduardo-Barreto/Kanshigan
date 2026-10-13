import { z } from 'zod'

import { clipEventSchema, reviewStatusSchema, robotSchema, roundSchema } from '@/shared/clip'

import { readJsonFile, writeFileAtomic } from './files'

const SIDECAR_VERSION = 1
const JSON_INDENT = 1

const sidecarSchema = z.object({
  version: z.literal(SIDECAR_VERSION),
  review: z.record(z.string(), reviewStatusSchema),
  identity: z.record(z.string(), z.array(robotSchema.nullable())),
  events: z.array(clipEventSchema),
  rounds: z.array(roundSchema),
})

/**
 * Everything the studio knows about a clip that a YOLO label file cannot hold: review
 * status, robot identity per box (parallel to the label lines), events and round limits.
 */
export type Sidecar = z.infer<typeof sidecarSchema>

/** A sidecar with nothing recorded yet. */
export function emptySidecar(): Sidecar {
  return { version: SIDECAR_VERSION, review: {}, identity: {}, events: [], rounds: [] }
}

/** Reads a clip's sidecar, or null when the studio has never written one for it. */
export async function readSidecar(path: string): Promise<Sidecar | null> {
  return readJsonFile(path, sidecarSchema)
}

/** Persists a clip's sidecar atomically. */
export async function writeSidecar(path: string, sidecar: Sidecar): Promise<void> {
  await writeFileAtomic(path, `${JSON.stringify(sidecar, null, JSON_INDENT)}\n`)
}
