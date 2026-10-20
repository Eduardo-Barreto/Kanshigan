import { mkdir, readdir, rename } from 'node:fs/promises'
import { dirname } from 'node:path'

import type { z } from 'zod'

/** Writes through a sibling temp file and renames, so a crash never leaves a half-written file. */
export async function writeFileAtomic(path: string, content: string): Promise<void> {
  await mkdir(dirname(path), { recursive: true })
  const temporaryPath = `${path}.${process.pid}.partial`
  await Bun.write(temporaryPath, content)
  await rename(temporaryPath, path)
}

/** Lists a directory, treating a missing one as empty: datasets are created lazily. */
export async function listDirectory(path: string): Promise<string[]> {
  try {
    return await readdir(path)
  } catch (error) {
    if (error instanceof Error && 'code' in error && error.code === 'ENOENT') return []
    throw error
  }
}

/** Reads and validates a JSON file, returning null when the file does not exist. */
export async function readJsonFile<Schema extends z.ZodType>(
  path: string,
  schema: Schema,
): Promise<z.infer<Schema> | null> {
  const file = Bun.file(path)
  if (!(await file.exists())) return null
  const parsed: unknown = JSON.parse(await file.text())
  return schema.parse(parsed)
}

/** Reads a text file, returning null when it does not exist. */
export async function readTextFile(path: string): Promise<string | null> {
  const file = Bun.file(path)
  return (await file.exists()) ? file.text() : null
}
