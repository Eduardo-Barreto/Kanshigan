import { describe, expect, test } from 'bun:test'

import { parseLabels, serializeLabels } from './labels'

describe('parseLabels', () => {
  test('reads one box per YOLO line', () => {
    expect(parseLabels('0 0.5 0.25 0.1 0.2\n0 0.75 0.5 0.3 0.4\n')).toEqual([
      { cx: 0.5, cy: 0.25, w: 0.1, h: 0.2 },
      { cx: 0.75, cy: 0.5, w: 0.3, h: 0.4 },
    ])
  })

  test('returns no boxes for an empty file', () => {
    expect(parseLabels('')).toEqual([])
  })

  test('skips blank lines', () => {
    expect(parseLabels('\n0 0.5 0.5 0.1 0.1\n\n   \n')).toEqual([
      { cx: 0.5, cy: 0.5, w: 0.1, h: 0.1 },
    ])
  })

  test('drops a line with too few fields', () => {
    expect(parseLabels('0 0.5 0.5 0.1\n0 0.5 0.5 0.1 0.1')).toEqual([
      { cx: 0.5, cy: 0.5, w: 0.1, h: 0.1 },
    ])
  })

  test('drops a line with a non-numeric coordinate', () => {
    expect(parseLabels('0 abc 0.5 0.1 0.1\n0 0.5 0.5 0.1 0.1')).toHaveLength(1)
  })

  test('drops a box with a coordinate outside the unit interval', () => {
    expect(parseLabels('0 1.2 0.5 0.1 0.1\n0 -0.1 0.5 0.1 0.1\n0 0.5 0.5 1.5 0.1')).toEqual([])
  })

  test('drops a box with zero width or height', () => {
    expect(parseLabels('0 0.5 0.5 0 0.1\n0 0.5 0.5 0.1 0')).toEqual([])
  })

  test('accepts boxes on the frame edge', () => {
    expect(parseLabels('0 0 1 1 1')).toEqual([{ cx: 0, cy: 1, w: 1, h: 1 }])
  })
})

describe('serializeLabels', () => {
  test('writes one line per box, each ending in a newline', () => {
    const text = serializeLabels([
      { cx: 0.5, cy: 0.5, w: 0.1, h: 0.1 },
      { cx: 0.2, cy: 0.3, w: 0.4, h: 0.5 },
    ])
    expect(text).toBe(
      '0 0.500000 0.500000 0.100000 0.100000\n0 0.200000 0.300000 0.400000 0.500000\n',
    )
  })

  test('rounds coordinates to six decimals', () => {
    expect(serializeLabels([{ cx: 1 / 3, cy: 2 / 3, w: 0.1234567, h: 0.9999999 }])).toBe(
      '0 0.333333 0.666667 0.123457 1.000000\n',
    )
  })

  test('returns an empty string for no boxes', () => {
    expect(serializeLabels([])).toBe('')
  })

  test('round-trips boxes through parse', () => {
    const boxes = [
      { cx: 0.123456, cy: 0.654321, w: 0.2, h: 0.3 },
      { cx: 0.9, cy: 0.1, w: 0.05, h: 0.07 },
    ]
    expect(parseLabels(serializeLabels(boxes))).toEqual(boxes)
  })
})
