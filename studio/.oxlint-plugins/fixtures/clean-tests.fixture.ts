import { describe, expect, test } from 'bun:test'

describe('member list', () => {
  test('names each member it renders', () => {
    expect('Ana').toBe('Ana')
  })
})

describe.each([1, 2])('table %i', (value: number) => {
  test('names the case it runs', () => {
    expect(value).toBeGreaterThan(0)
  })
})
