import { describe, expect, it } from 'bun:test'

describe('table form', () => {
  it.each([1, 2])('should handle %i', (value: number) => {
    expect(value).toBeGreaterThan(0)
  })
})
