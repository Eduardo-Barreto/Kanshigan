import { serveStatic } from 'hono/bun'

import { MAX_UPLOAD_BYTES, createApp } from './app'
import { loadConfig } from './config'
import { log } from './log'

const LOOPBACK_HOST = '127.0.0.1'

const config = loadConfig(process.env)
const app = createApp(config)

app.use('*', serveStatic({ root: config.staticRoot }))
app.get('*', serveStatic({ path: 'index.html', root: config.staticRoot }))

const server = Bun.serve({
  port: config.port,
  hostname: LOOPBACK_HOST,
  maxRequestBodySize: MAX_UPLOAD_BYTES,
  fetch: app.fetch,
})
log.info('server.listening', {
  url: `http://${LOOPBACK_HOST}:${server.port}`,
  dataRoot: config.dataRoot,
})
