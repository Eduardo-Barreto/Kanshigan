import { resolve } from 'node:path'

import { z } from 'zod'

const DEFAULT_PORT = 4310
const MIN_PASSWORD_LENGTH = 12
const REPO_ROOT = resolve(import.meta.dir, '../../..')

const environmentSchema = z.object({
  STUDIO_PORT: z.coerce.number().int().positive().default(DEFAULT_PORT),
  STUDIO_DATA_ROOT: z.string().default(resolve(REPO_ROOT, 'data')),
  STUDIO_REPO_ROOT: z.string().default(REPO_ROOT),
  STUDIO_PASSWORD: z
    .string()
    .min(MIN_PASSWORD_LENGTH)
    .or(z.literal('').transform(() => undefined))
    .optional(),
  STUDIO_STATIC_ROOT: z.string().default(resolve(import.meta.dir, '../../dist')),
})

export type StudioConfig = {
  port: number
  dataRoot: string
  repoRoot: string
  staticRoot: string
  password: string | undefined
  paths: {
    annotations: string
    rawVideos: string
    clipVideos: string
    samInput: string
    frameCache: string
    jobLogs: string
  }
}

/** Reads the server configuration from the environment, failing fast on a malformed value. */
export function loadConfig(environment: Record<string, string | undefined>): StudioConfig {
  const parsed = environmentSchema.parse(environment)
  const dataRoot = resolve(parsed.STUDIO_DATA_ROOT)
  return {
    port: parsed.STUDIO_PORT,
    dataRoot,
    repoRoot: resolve(parsed.STUDIO_REPO_ROOT),
    staticRoot: resolve(parsed.STUDIO_STATIC_ROOT),
    password: parsed.STUDIO_PASSWORD,
    paths: {
      annotations: resolve(dataRoot, 'annotations'),
      rawVideos: resolve(dataRoot, 'raw'),
      clipVideos: resolve(dataRoot, 'processed', 'clips'),
      samInput: resolve(dataRoot, 'processed', 'sam_input'),
      frameCache: resolve(dataRoot, 'studio', 'frame-cache'),
      jobLogs: resolve(dataRoot, 'studio', 'jobs'),
    },
  }
}
