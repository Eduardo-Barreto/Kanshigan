import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { GlobalRegistrator } from '@happy-dom/global-registrator'

import { sessionSchema } from '@/shared/auth'

import { createApp } from './app'
import { loadConfig } from './config'
import { createJobQueue } from './jobs'

const PASSWORD = 'a-long-test-password'
const MAX_FAILED_LOGINS = 5

let dataRoot = ''

// happy-dom's Headers hides Set-Cookie, so the session cookie is only visible with native globals.
beforeAll(() => GlobalRegistrator.unregister())
afterAll(() => GlobalRegistrator.register())

beforeEach(async () => {
  dataRoot = await mkdtemp(join(tmpdir(), 'studio-auth-'))
})

afterEach(async () => {
  await rm(dataRoot, { recursive: true, force: true })
})

function appWith(password: string | undefined) {
  const config = loadConfig({ STUDIO_DATA_ROOT: dataRoot, STUDIO_PASSWORD: password })
  return createApp(config, createJobQueue(join(dataRoot, 'jobs')))
}

const login = (app: ReturnType<typeof appWith>, password: string) =>
  app.request('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ password }),
  })

describe('without a configured password', () => {
  test('reports an open session', async () => {
    const response = await appWith(undefined).request('/api/auth/session')
    expect(sessionSchema.parse(await response.json())).toEqual({
      authenticated: true,
      via: 'open',
    })
  })

  test('serves the library', async () => {
    const response = await appWith(undefined).request('/api/library')
    expect(response.status).toBe(200)
  })
})

describe('with a configured password', () => {
  test('rejects the library without credentials', async () => {
    const response = await appWith(PASSWORD).request('/api/library')
    expect(response.status).toBe(401)
  })

  test('reports an unauthenticated session without credentials', async () => {
    const response = await appWith(PASSWORD).request('/api/auth/session')
    expect(response.status).toBe(200)
    expect(sessionSchema.parse(await response.json())).toEqual({ authenticated: false, via: null })
  })

  test('lets in a request carrying the Tailscale identity header', async () => {
    const response = await appWith(PASSWORD).request('/api/library', {
      headers: { 'Tailscale-User-Login': 'someone@example.com' },
    })
    expect(response.status).toBe(200)
  })

  test('answers 401 to a wrong password', async () => {
    const response = await login(appWith(PASSWORD), 'not-the-password')
    expect(response.status).toBe(401)
    expect(response.headers.getSetCookie()).toEqual([])
  })

  test('starts a cookie session on the right password', async () => {
    const app = appWith(PASSWORD)
    const response = await login(app, PASSWORD)
    expect(response.status).toBe(200)
    const cookie = response.headers.getSetCookie()[0]
    expect(cookie).toContain('studio_session=')

    const followUp = await app.request('/api/library', {
      headers: { Cookie: cookie?.split(';')[0] ?? '' },
    })
    expect(followUp.status).toBe(200)
  })

  test('rejects a forged session cookie', async () => {
    const response = await appWith(PASSWORD).request('/api/library', {
      headers: { Cookie: 'studio_session=granted' },
    })
    expect(response.status).toBe(401)
  })

  test('locks out logins after repeated wrong attempts even with the right password', async () => {
    const app = appWith(PASSWORD)
    for (let attempt = 0; attempt < MAX_FAILED_LOGINS; attempt += 1) {
      // oxlint-disable-next-line no-await-in-loop -- attempts must be counted in order
      expect((await login(app, 'wrong-password')).status).toBe(401)
    }
    expect((await login(app, PASSWORD)).status).toBe(429)
  })
})
