import { deck } from '@nimbus/deck/vite'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [deck({ name: 'Kanshigan', short: 'Kanshigan' })],
})
