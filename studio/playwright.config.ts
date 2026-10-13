import { defineConfig, devices } from '@playwright/test'

const port = 5183
const baseURL = `http://127.0.0.1:${port}`

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI ? 'github' : 'html',
  use: { baseURL, trace: 'on-first-retry' },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    command: `bun run build && bun scripts/e2e-server.ts ${port}`,
    url: `${baseURL}/api/auth/session`,
    reuseExistingServer: false,
  },
})
