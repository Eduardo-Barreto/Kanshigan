import { afterAll, beforeAll, describe, expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { annotateRequestSchema } from '@/shared/ingest'

import { createApp } from './app'
import { loadConfig } from './config'
import { datasetProfile } from './datasets'
import { canonicalYoutubeUrl } from './ingest'
import { createJobQueue } from './jobs'
import { Problem } from './problem'

const PASSWORD = 'a-long-test-password'
const FUNNEL = { 'Tailscale-Funnel-Request': '?1', 'X-Forwarded-For': '203.0.113.7' }
const JSON_HEADERS = { 'Content-Type': 'application/json' }

let dataRoot = ''

beforeAll(async () => {
  dataRoot = await mkdtemp(join(tmpdir(), 'studio-hardening-'))
})

afterAll(async () => {
  await rm(dataRoot, { recursive: true, force: true })
})

function appWith(password?: string) {
  const config = loadConfig({ STUDIO_DATA_ROOT: dataRoot, STUDIO_PASSWORD: password })
  return createApp(config, createJobQueue(join(dataRoot, 'jobs')))
}

function loginFrom(app: ReturnType<typeof appWith>, client: string, password: string) {
  return app.request('/api/auth/login', {
    method: 'POST',
    headers: { ...JSON_HEADERS, 'X-Forwarded-For': client },
    body: JSON.stringify({ password }),
  })
}

describe('requests arriving through the Funnel', () => {
  test('are refused when the server has no password, so a forgotten Funnel serves nothing', async () => {
    const response = await appWith().request('/api/library', { headers: FUNNEL })
    expect(response.status).toBe(401)
  })

  test('are not trusted for carrying a tailnet identity header', async () => {
    const response = await appWith(PASSWORD).request('/api/library', {
      headers: { ...FUNNEL, 'Tailscale-User-Login': 'forged@example.com' },
    })
    expect(response.status).toBe(401)
  })
})

describe('host check', () => {
  test('refuses a hostname that is neither loopback nor a Tailscale name', async () => {
    const response = await appWith().request('http://attacker.example:4310/api/library')
    expect(response.status).toBe(403)
  })

  test('accepts a MagicDNS name with a port', async () => {
    const response = await appWith().request('https://machine.tail1234.ts.net:8443/api/library')
    expect(response.status).toBe(200)
  })
})

describe('login lockout', () => {
  test('locks out the client that keeps guessing and nobody else', async () => {
    const app = appWith(PASSWORD)
    for (let attempt = 0; attempt < 5; attempt += 1) {
      // oxlint-disable-next-line no-await-in-loop -- attempts must land in order
      await loginFrom(app, '203.0.113.7', 'wrong')
    }

    expect((await loginFrom(app, '203.0.113.7', PASSWORD)).status).toBe(429)
    expect((await loginFrom(app, '198.51.100.9', PASSWORD)).status).toBe(200)
  })
})

describe('canonicalYoutubeUrl', () => {
  test.each([
    ['https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123', 'dQw4w9WgXcQ'],
    ['https://youtu.be/dQw4w9WgXcQ?t=10', 'dQw4w9WgXcQ'],
    ['https://www.youtube.com/shorts/dQw4w9WgXcQ', 'dQw4w9WgXcQ'],
  ])('reduces %s to its video id', (input, videoId) => {
    expect(canonicalYoutubeUrl(input)).toBe(`https://www.youtube.com/watch?v=${videoId}`)
  })

  test.each([
    ['a host smuggled behind a userinfo separator', 'http://youtube.com\\@127.0.0.1:8080/x.mp4'],
    ['a non-https scheme', 'file://youtube.com/etc/passwd'],
    ['a channel page', 'https://www.youtube.com/@NASA/videos'],
    ['a playlist without a video', 'https://www.youtube.com/playlist?list=PL123'],
    ['another site', 'https://example.com/watch?v=dQw4w9WgXcQ'],
  ])('rejects %s', (_case, input) => {
    expect(() => canonicalYoutubeUrl(input)).toThrow(Problem)
  })
})

describe('job queue', () => {
  test('refuses new jobs once too many are waiting', () => {
    const queue = createJobQueue(join(dataRoot, 'jobs-cap'))
    const never = { label: 'wait', run: () => new Promise<void>(() => undefined) }
    const spec = { kind: 'segment' as const, title: 'wait', video: null, steps: [never] }
    for (let queued = 0; queued < 20; queued += 1) queue.enqueue(spec)

    expect(() => queue.enqueue(spec)).toThrow(Problem)
  })
})

describe('datasetProfile', () => {
  test('treats a name inherited from Object.prototype as an unknown dataset', () => {
    expect(datasetProfile('constructor')).toMatchObject({ role: 'other', label: 'constructor' })
  })
})

describe('annotate request', () => {
  test('rejects a window too long to decode into memory', () => {
    const request = {
      source: 'br',
      video: 'match',
      dataset: 'gold_candidates',
      clipPrefix: 'match',
      windows: [{ start: 0, end: 7200 }],
      scoreThreshold: null,
    }
    expect(annotateRequestSchema.safeParse(request).success).toBe(false)
  })
})
