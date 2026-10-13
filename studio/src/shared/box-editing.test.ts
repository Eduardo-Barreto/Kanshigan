import { describe, expect, test } from 'bun:test'

import {
  boxFromCorners,
  containsPoint,
  cornersOf,
  dragCorner,
  moveBox,
  resizeBox,
} from './box-editing'
import type { Box } from './clip'

const box: Box = { cx: 0.5, cy: 0.5, w: 0.2, h: 0.1, robot: 'A' }

describe('boxFromCorners', () => {
  test('builds the same box whichever corner the drag starts from', () => {
    const forward = boxFromCorners({ x: 0.1, y: 0.2 }, { x: 0.3, y: 0.6 })
    const backward = boxFromCorners({ x: 0.3, y: 0.6 }, { x: 0.1, y: 0.2 })
    expect(forward).toEqual(backward)
    expect(forward?.cx).toBeCloseTo(0.2)
    expect(forward?.h).toBeCloseTo(0.4)
  })

  test('clips a drag that ends outside the frame', () => {
    const clipped = boxFromCorners({ x: 0.8, y: 0.8 }, { x: 1.4, y: 1.2 })
    expect(clipped?.cx).toBeCloseTo(0.9)
    expect(clipped?.w).toBeCloseTo(0.2)
  })

  test('rejects a click that never became a drag', () => {
    expect(boxFromCorners({ x: 0.5, y: 0.5 }, { x: 0.501, y: 0.9 })).toBeNull()
  })
})

describe('moveBox', () => {
  test('stops at the frame edge without shrinking the box', () => {
    const moved = moveBox(box, 0.9, 0)
    expect(moved.cx).toBeCloseTo(0.9)
    expect(moved.w).toBe(box.w)
  })
})

describe('resizeBox', () => {
  test('grows around the centre', () => {
    const grown = resizeBox(box, 0.1, 0.1)
    expect(grown.cx).toBe(box.cx)
    expect(grown.w).toBeCloseTo(0.3)
    expect(grown.h).toBeCloseTo(0.2)
  })

  test('never collapses below the minimum size', () => {
    expect(resizeBox(box, -1, -1).w).toBeGreaterThan(0)
  })

  test('cannot grow past the frame', () => {
    const nearEdge: Box = { ...box, cx: 0.95 }
    expect(resizeBox(nearEdge, 1, 0).w).toBeCloseTo(0.1)
  })
})

describe('dragCorner', () => {
  test('anchors the opposite corner and keeps the robot', () => {
    const dragged = dragCorner(box, 3, { x: 0.8, y: 0.7 })
    const [topLeft] = cornersOf(dragged)
    expect(topLeft.x).toBeCloseTo(0.4)
    expect(topLeft.y).toBeCloseTo(0.45)
    expect(dragged.robot).toBe('A')
  })
})

describe('containsPoint', () => {
  test('includes points inside and excludes points outside', () => {
    expect(containsPoint(box, { x: 0.59, y: 0.54 })).toBe(true)
    expect(containsPoint(box, { x: 0.61, y: 0.5 })).toBe(false)
  })
})
