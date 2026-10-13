#!/usr/bin/env bun
/**
 * Starts the studio for Playwright against a throwaway copy of the fixture library, so a
 * test that edits annotations never touches the committed fixtures or the real data.
 *
 * Usage: bun scripts/e2e-server.ts <port>
 */

import { cp, mkdtemp } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'

const fixtureLibrary = resolve(import.meta.dir, '../e2e/fixtures/library')
const dataRoot = await mkdtemp(join(tmpdir(), 'kanshigan-studio-e2e-'))
await cp(fixtureLibrary, dataRoot, { recursive: true })

const server = Bun.spawn(['bun', resolve(import.meta.dir, '../src/server/main.ts')], {
  env: {
    ...process.env,
    STUDIO_DATA_ROOT: dataRoot,
    STUDIO_PORT: process.argv[2],
    STUDIO_PASSWORD: '',
  },
  stdout: 'inherit',
  stderr: 'inherit',
})
process.exit(await server.exited)
