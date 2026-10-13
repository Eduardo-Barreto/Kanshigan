import { describe, expect, test } from 'bun:test'

import { backendsOnPort, refusal } from './expose.ts'

const studio = 'http://127.0.0.1:4310'
const status = {
  Web: {
    'host.ts.net:443': { Handlers: { '/': { Proxy: 'http://127.0.0.1:3773' } } },
    'host.ts.net:8443': { Handlers: { '/': { Proxy: studio } } },
  },
}

describe('backendsOnPort', () => {
  test('lists only the backends proxied on the asked port', () => {
    expect(backendsOnPort(status, '8443')).toEqual([studio])
    expect(backendsOnPort(status, '443')).toEqual(['http://127.0.0.1:3773'])
  })

  test('finds nothing when tailscale serves nothing', () => {
    expect(backendsOnPort({}, '8443')).toEqual([])
  })
})

describe('refusal', () => {
  test('refuses to take over a port that proxies another app', () => {
    const reason = refusal({
      mode: 'tailnet',
      occupants: ['http://127.0.0.1:3773'],
      backend: studio,
      sessionVia: 'open',
    })
    expect(reason).toContain('http://127.0.0.1:3773')
  })

  test('refuses to open the Funnel while the server has no password', () => {
    expect(
      refusal({ mode: 'funnel', occupants: [], backend: studio, sessionVia: 'open' }),
    ).toContain('STUDIO_PASSWORD')
  })

  test('opens the Funnel when the server asks for a login', () => {
    expect(
      refusal({ mode: 'funnel', occupants: [studio], backend: studio, sessionVia: null }),
    ).toBeNull()
  })

  test('serves the tailnet without a password', () => {
    expect(
      refusal({ mode: 'tailnet', occupants: [], backend: studio, sessionVia: 'open' }),
    ).toBeNull()
  })
})
