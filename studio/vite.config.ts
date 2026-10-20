import path from 'node:path'

import tailwindcss from '@tailwindcss/vite'
import { tanstackRouter } from '@tanstack/router-plugin/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const base = process.env.BASE_PATH ?? '/'
const apiOrigin = `http://127.0.0.1:${process.env.STUDIO_PORT ?? '4310'}`

export default defineConfig(({ command }) => ({
  base,
  // Dev-only LAN exposure; never applied to `vite build`.
  server:
    command === 'serve'
      ? {
          // Loopback only, and the proxy drops the tailnet identity header: the API trusts
          // that header because only Tailscale can set it, which a LAN client could forge here.
          host: '127.0.0.1',
          proxy: {
            '/api': {
              target: apiOrigin,
              configure: (proxy) =>
                proxy.on('proxyReq', (request) => request.removeHeader('Tailscale-User-Login')),
            },
          },
        }
      : undefined,
  plugins: [tanstackRouter({ target: 'react', autoCodeSplitting: true }), react(), tailwindcss()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
}))
