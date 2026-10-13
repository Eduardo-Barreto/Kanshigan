import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { z } from 'zod'

import { clipDetailSchema, librarySchema } from '@/shared/clip'

import { createApp } from './app'
import { loadConfig } from './config'
import { createJobQueue } from './jobs'

const errorSchema = z.object({ error: z.string() })

let dataRoot = ''
let app: ReturnType<typeof createApp>

beforeEach(async () => {
  dataRoot = await mkdtemp(join(tmpdir(), 'studio-app-'))
  const config = loadConfig({ STUDIO_DATA_ROOT: dataRoot })
  app = createApp(config, createJobQueue(join(dataRoot, 'jobs')))
  const stem = join(dataRoot, 'annotations', 'train')
  await Bun.write(join(stem, 'images', 'match_0000.jpg'), new Uint8Array([0xff, 0xd8]))
  await Bun.write(join(stem, 'labels', 'match_0000.txt'), '0 0.5 0.5 0.1 0.2\n')
})

afterEach(async () => {
  await rm(dataRoot, { recursive: true, force: true })
})

const putFrames = (body: object) =>
  app.request('/api/clips/train/match/frames', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

const frameSave = (width: number) => ({
  frames: [
    {
      index: 0,
      review: 'approved',
      boxes: [{ cx: 0.4, cy: 0.4, w: width, h: 0.2, robot: 'A' }],
    },
  ],
})

describe('GET /api/library', () => {
  test('returns the datasets with their clips', async () => {
    const response = await app.request('/api/library')
    expect(response.status).toBe(200)
    const library = librarySchema.parse(await response.json())
    expect(library.datasets.map((dataset) => dataset.dataset)).toEqual(['train'])
    expect(library.datasets[0]?.clips.map((clip) => clip.clip)).toEqual(['match'])
  })
})

describe('path parameters', () => {
  test('answers 400 to a path-traversal dataset name', async () => {
    const response = await app.request('/api/clips/..%2Fetc/match')
    expect(response.status).toBe(400)
  })

  test('answers 400 to a clip name with an invalid character', async () => {
    const response = await app.request('/api/clips/train/ma%20tch')
    expect(response.status).toBe(400)
  })

  test('answers 400 to a clip name that starts with a dot', async () => {
    const response = await app.request('/api/clips/train/.hidden')
    expect(response.status).toBe(400)
  })
})

describe('GET /api/clips/:dataset/:clip', () => {
  test('returns the clip detail', async () => {
    const response = await app.request('/api/clips/train/match')
    expect(response.status).toBe(200)
    const detail = clipDetailSchema.parse(await response.json())
    expect(detail.frames).toHaveLength(1)
  })

  test('answers 404 with a JSON error for a missing clip', async () => {
    const response = await app.request('/api/clips/train/ghost')
    expect(response.status).toBe(404)
    expect(errorSchema.parse(await response.json()).error).toContain('ghost')
  })
})

describe('PUT /api/clips/:dataset/:clip/frames', () => {
  test('answers 400 to a box with zero width', async () => {
    const response = await putFrames(frameSave(0))
    expect(response.status).toBe(400)
  })

  test('answers 400 to an empty frame list', async () => {
    const response = await putFrames({ frames: [] })
    expect(response.status).toBe(400)
  })

  test('persists a valid save and returns the updated clip', async () => {
    const response = await putFrames(frameSave(0.3))
    expect(response.status).toBe(200)
    const detail = clipDetailSchema.parse(await response.json())
    expect(detail.frames[0]).toMatchObject({ review: 'approved' })
    expect(detail.frames[0]?.boxes).toEqual([{ cx: 0.4, cy: 0.4, w: 0.3, h: 0.2, robot: 'A' }])

    const reread = clipDetailSchema.parse(
      await (await app.request('/api/clips/train/match')).json(),
    )
    expect(reread.frames[0]?.boxes[0]?.robot).toBe('A')
  })
})

describe('POST /api/annotations', () => {
  test('answers 422 when the target dataset is frozen', async () => {
    const response = await app.request('/api/annotations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        source: 'br',
        video: 'match',
        dataset: 'train',
        clipPrefix: 'round',
        windows: [{ start: 0, end: 5 }],
        scoreThreshold: null,
      }),
    })
    expect(response.status).toBe(422)
    expect(errorSchema.parse(await response.json()).error).toContain('train')
  })
})
