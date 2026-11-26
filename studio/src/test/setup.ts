import { afterEach, expect } from 'bun:test'

import { GlobalRegistrator } from '@happy-dom/global-registrator'
import * as matchers from '@testing-library/jest-dom/matchers'

// happy-dom's Headers hides Set-Cookie and its fetch classes differ from Bun's; server tests
// need the native ones, and swapping globals per file left Testing Library bound to a stale
// document for every component test that ran afterwards.
const nativeFetchApi = { fetch, Headers, Request, Response }
GlobalRegistrator.register()
Object.assign(globalThis, nativeFetchApi)
expect.extend(matchers)

const { cleanup } = await import('@testing-library/react')
afterEach(() => cleanup())
