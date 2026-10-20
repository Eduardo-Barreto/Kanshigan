import { describe, expect, test } from 'bun:test'

import type { Box } from '@/shared/clip'

import { beginDrag, continueDrag, endDrag } from './stage-drag'

const left: Box = { cx: 0.3, cy: 0.5, w: 0.2, h: 0.2, robot: 'A' }
const right: Box = { cx: 0.7, cy: 0.5, w: 0.2, h: 0.2, robot: 'B' }
const tolerance = { x: 0.01, y: 0.01 }

describe('beginDrag', () => {
  test('moves the box under the pointer', () => {
    expect(beginDrag([left, right], null, { x: 0.7, y: 0.5 }, tolerance)).toMatchObject({
      kind: 'move',
      boxIndex: 1,
    })
  })

  test('grabs a corner of the selected box before anything else', () => {
    expect(beginDrag([left, right], 0, { x: 0.4, y: 0.6 }, tolerance)).toEqual({
      kind: 'corner',
      boxIndex: 0,
      cornerIndex: 3,
    })
  })

  test('draws when the press lands on empty frame', () => {
    expect(beginDrag([left, right], 0, { x: 0.5, y: 0.1 }, tolerance).kind).toBe('draw')
  })
})

describe('continueDrag', () => {
  test('moves a box by the pointer travel since the press', () => {
    const drag = beginDrag([left, right], null, { x: 0.3, y: 0.5 }, tolerance)
    const result = continueDrag(drag, [left, right], { x: 0.35, y: 0.45 })
    expect(result.boxes[0]?.cx).toBeCloseTo(0.35)
    expect(result.boxes[0]?.cy).toBeCloseTo(0.45)
    expect(result.selected).toBe(0)
  })

  test('previews a drawn box without adding it yet', () => {
    const drag = beginDrag([left], null, { x: 0.6, y: 0.1 }, tolerance)
    const result = continueDrag(drag, [left], { x: 0.8, y: 0.3 })
    expect(result.boxes).toHaveLength(1)
    expect(result.preview?.w).toBeCloseTo(0.2)
  })
})

describe('endDrag', () => {
  test('adds the drawn box and selects it', () => {
    const drag = beginDrag([left], null, { x: 0.6, y: 0.1 }, tolerance)
    const result = endDrag(drag, [left], { x: 0.8, y: 0.3 })
    expect(result.boxes).toHaveLength(2)
    expect(result.selected).toBe(1)
  })

  test('adds nothing for a press that never became a drag', () => {
    const drag = beginDrag([left], null, { x: 0.6, y: 0.1 }, tolerance)
    expect(endDrag(drag, [left], { x: 0.6, y: 0.1 }).boxes).toHaveLength(1)
  })
})
