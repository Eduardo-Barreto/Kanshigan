import { describe, expect, test } from 'bun:test'

import { resolveRange } from './range'

const FILE_SIZE = 1000

describe('resolveRange', () => {
  test('serves the whole file when there is no header', () => {
    expect(resolveRange(undefined, FILE_SIZE)).toBeNull()
  })

  test('serves the whole file when the header is malformed', () => {
    expect(resolveRange('items=0-10', FILE_SIZE)).toBeNull()
    expect(resolveRange('bytes=abc', FILE_SIZE)).toBeNull()
    expect(resolveRange('bytes=-', FILE_SIZE)).toBeNull()
  })

  test('resolves an explicit inclusive range', () => {
    expect(resolveRange('bytes=0-99', FILE_SIZE)).toEqual({ start: 0, end: 99 })
  })

  test('resolves an open-ended range to the last byte', () => {
    expect(resolveRange('bytes=100-', FILE_SIZE)).toEqual({ start: 100, end: 999 })
  })

  test('resolves a suffix range to the final bytes', () => {
    expect(resolveRange('bytes=-50', FILE_SIZE)).toEqual({ start: 950, end: 999 })
  })

  test('clamps a suffix longer than the file to the whole file', () => {
    expect(resolveRange('bytes=-5000', FILE_SIZE)).toEqual({ start: 0, end: 999 })
  })

  test('clamps an end past the file size to the last byte', () => {
    expect(resolveRange('bytes=900-5000', FILE_SIZE)).toEqual({ start: 900, end: 999 })
  })

  test('reports unsatisfiable when the start is past the end of the file', () => {
    expect(resolveRange('bytes=1000-', FILE_SIZE)).toBe('unsatisfiable')
    expect(resolveRange('bytes=2000-2100', FILE_SIZE)).toBe('unsatisfiable')
  })

  test('reports unsatisfiable when the start is after the end', () => {
    expect(resolveRange('bytes=50-10', FILE_SIZE)).toBe('unsatisfiable')
  })

  test('reports unsatisfiable for a suffix range on an empty file', () => {
    expect(resolveRange('bytes=-10', 0)).toBe('unsatisfiable')
  })
})
