#!/usr/bin/env bun
/**
 * Puts the studio on the tailnet or on the public internet through Tailscale, and takes it
 * back off. The server itself only ever listens on loopback.
 *
 * Usage: bun scripts/expose.ts <tailnet|funnel|off>
 */

import { z } from 'zod'

const DEFAULT_STUDIO_PORT = '4310'
const DEFAULT_PUBLIC_PORT = '8443'
const MODES = ['tailnet', 'funnel', 'off'] as const

const serveStatusSchema = z.looseObject({
  Web: z
    .record(
      z.string(),
      z.looseObject({
        Handlers: z.record(z.string(), z.looseObject({ Proxy: z.string().optional() })).optional(),
      }),
    )
    .optional(),
})
const sessionSchema = z.object({ via: z.string().nullable() })

type ServeStatus = z.infer<typeof serveStatusSchema>

/** Backends already proxied on a public port, read from `tailscale serve status --json`. */
export function backendsOnPort(status: ServeStatus, publicPort: string): string[] {
  return Object.entries(status.Web ?? {})
    .filter(([hostAndPort]) => hostAndPort.endsWith(`:${publicPort}`))
    .flatMap(([, site]) => Object.values(site.Handlers ?? {}))
    .map((handler) => handler.Proxy)
    .filter((proxy) => proxy !== undefined)
}

/**
 * Explains why exposing must not go ahead, or returns null when it may. A port that already
 * proxies something else is never taken over: `tailscale serve` would silently replace it.
 */
export function refusal(input: {
  mode: (typeof MODES)[number]
  occupants: string[]
  backend: string
  sessionVia: string | null
}): string | null {
  const foreign = input.occupants.filter((occupant) => occupant !== input.backend)
  if (foreign.length > 0) {
    return `a porta já serve ${foreign.join(', ')}; escolha outra com STUDIO_PUBLIC_PORT`
  }
  if (input.mode === 'funnel' && input.sessionVia === 'open') {
    return 'o servidor está sem senha; defina STUDIO_PASSWORD no .env e reinicie antes de abrir o Funnel'
  }
  return null
}

function tailscale(args: string[]): string {
  const result = Bun.spawnSync(['tailscale', ...args], { stdout: 'pipe', stderr: 'pipe' })
  if (result.exitCode !== 0) throw new Error(result.stderr.toString().trim())
  return result.stdout.toString()
}

async function sessionVia(backend: string): Promise<string | null> {
  const response = await fetch(`${backend}/api/auth/session`)
  const body: unknown = await response.json()
  return sessionSchema.parse(body).via
}

async function main() {
  const mode = z.enum(MODES).parse(process.argv[2])
  const publicPort = process.env.STUDIO_PUBLIC_PORT ?? DEFAULT_PUBLIC_PORT
  const backend = `http://127.0.0.1:${process.env.STUDIO_PORT ?? DEFAULT_STUDIO_PORT}`
  const statusJson: unknown = JSON.parse(tailscale(['serve', 'status', '--json']) || '{}')
  const occupants = backendsOnPort(serveStatusSchema.parse(statusJson), publicPort)

  if (mode === 'off') {
    if (occupants.includes(backend)) tailscale(['serve', `--https=${publicPort}`, 'off'])
    console.log(`Studio fora da porta ${publicPort}.`)
    return
  }
  const reason = refusal({ mode, occupants, backend, sessionVia: await sessionVia(backend) })
  if (reason !== null) {
    console.error(`Não expus: ${reason}.`)
    process.exit(1)
  }
  if (occupants.length > 0) tailscale(['serve', `--https=${publicPort}`, 'off'])
  tailscale([mode === 'funnel' ? 'funnel' : 'serve', '--bg', `--https=${publicPort}`, backend])
  console.log(tailscale([mode === 'funnel' ? 'funnel' : 'serve', 'status']).trim())
}

if (import.meta.main) await main()
