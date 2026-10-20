import { createHash, randomBytes, timingSafeEqual } from 'node:crypto'

import type { Context, MiddlewareHandler } from 'hono'
import { deleteCookie, getSignedCookie, setSignedCookie } from 'hono/cookie'

import type { Session } from '@/shared/auth'

import { Problem } from './problem'

const SESSION_COOKIE = 'studio_session'
const SESSION_MAX_AGE_SECONDS = 604_800
const MS_PER_SECOND = 1000
const SECRET_BYTES = 32
const TAILNET_IDENTITY_HEADER = 'Tailscale-User-Login'
const FUNNEL_HEADER = 'Tailscale-Funnel-Request'
const FORWARDED_FOR_HEADER = 'X-Forwarded-For'
const LOCAL_CLIENT = 'local'
const MAX_FAILED_LOGINS_PER_CLIENT = 5
const MAX_FAILED_LOGINS_OVERALL = 100
const LOCKOUT_WINDOW_MS = 60_000
const HTTP_UNAUTHORIZED = 401
const HTTP_FORBIDDEN = 403
const HTTP_TOO_MANY_REQUESTS = 429

// Loopback names for direct use and the Vite proxy, and MagicDNS names for Tailscale Serve
// and Funnel. A page that reaches the server through DNS rebinding carries its own hostname.
const ALLOWED_HOST = /^(?:localhost|127\.0\.0\.1|(?:[a-z0-9-]+\.)+ts\.net)(?::\d+)?$/i

function digest(text: string): Buffer {
  return createHash('sha256').update(text).digest()
}

function isFromFunnel(context: Context): boolean {
  return context.req.header(FUNNEL_HEADER) !== undefined
}

function isFromTailnet(context: Context): boolean {
  return Boolean(context.req.header(TAILNET_IDENTITY_HEADER)) && !isFromFunnel(context)
}

function clientOf(context: Context): string {
  return context.req.header(FORWARDED_FOR_HEADER)?.split(',').at(-1)?.trim() ?? LOCAL_CLIENT
}

/** Refuses a request whose Host is neither loopback nor a Tailscale name: the mark of DNS rebinding. */
export const hostGuard: MiddlewareHandler = async (context, next) => {
  if (!ALLOWED_HOST.test(new URL(context.req.url).host)) {
    throw new Problem(HTTP_FORBIDDEN, 'host não permitido')
  }
  await next()
}

/**
 * Access control for a server that only listens on loopback and is reached through
 * Tailscale. What the proxy sends was checked against real traffic: a tailnet request
 * carries `Tailscale-User-Login` (which Tailscale strips from anything a client sends), and
 * a Funnel request carries `Tailscale-Funnel-Request` and no identity.
 *
 * Without a password the API is open to this machine and the tailnet, and closed to the
 * Funnel, so a Funnel left on never serves an unprotected server. With a password, a
 * tailnet identity or a login session is required.
 */
export function createAuth(password: string | undefined) {
  const secret = randomBytes(SECRET_BYTES).toString('hex')
  const failedLogins = new Map<string, number[]>()

  const hasSession = async (context: Context): Promise<boolean> => {
    const expiresAt = Number(await getSignedCookie(context, secret, SESSION_COOKIE))
    return Number.isFinite(expiresAt) && expiresAt > Date.now()
  }

  const sessionOf = async (context: Context): Promise<Session> => {
    if (isFromTailnet(context)) return { authenticated: true, via: 'tailnet' }
    if (password === undefined) {
      return isFromFunnel(context)
        ? { authenticated: false, via: null }
        : { authenticated: true, via: 'open' }
    }
    return (await hasSession(context))
      ? { authenticated: true, via: 'password' }
      : { authenticated: false, via: null }
  }

  const recentFailures = (client: string): number[] => {
    const windowStart = Date.now() - LOCKOUT_WINDOW_MS
    const recent = (failedLogins.get(client) ?? []).filter((failedAt) => failedAt > windowStart)
    if (recent.length === 0) failedLogins.delete(client)
    else failedLogins.set(client, recent)
    return recent
  }

  const assertNotLockedOut = (client: string) => {
    const overall = [...failedLogins.keys()].reduce(
      (total, known) => total + recentFailures(known).length,
      0,
    )
    const isLocked =
      recentFailures(client).length >= MAX_FAILED_LOGINS_PER_CLIENT ||
      overall >= MAX_FAILED_LOGINS_OVERALL
    if (isLocked) throw new Problem(HTTP_TOO_MANY_REQUESTS, 'muitas tentativas; espere um minuto')
  }

  return {
    sessionOf,
    /** Rejects API calls from a browser that is neither on the tailnet nor logged in. */
    guard: (async (context, next) => {
      const session = await sessionOf(context)
      if (!session.authenticated) throw new Problem(HTTP_UNAUTHORIZED, 'entre com a senha')
      await next()
    }) satisfies MiddlewareHandler,
    /**
     * Checks a password in constant time and, when right, starts a cookie session that the
     * server itself expires. Wrong attempts are counted per client address, so one client
     * guessing cannot lock the others out.
     */
    async login(context: Context, attempt: string): Promise<Session> {
      const client = clientOf(context)
      assertNotLockedOut(client)
      const isRight = password !== undefined && timingSafeEqual(digest(attempt), digest(password))
      if (!isRight) {
        failedLogins.set(client, [...recentFailures(client), Date.now()])
        throw new Problem(HTTP_UNAUTHORIZED, 'senha incorreta')
      }
      const expiresAt = Date.now() + SESSION_MAX_AGE_SECONDS * MS_PER_SECOND
      await setSignedCookie(context, SESSION_COOKIE, String(expiresAt), secret, {
        httpOnly: true,
        secure: true,
        sameSite: 'Strict',
        path: '/',
        maxAge: SESSION_MAX_AGE_SECONDS,
      })
      return { authenticated: true, via: 'password' }
    },
    logout(context: Context): Session {
      deleteCookie(context, SESSION_COOKIE, { path: '/' })
      return { authenticated: false, via: null }
    },
  }
}
