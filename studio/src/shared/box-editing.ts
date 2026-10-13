import type { Box, Robot } from './clip'

const MIN_BOX_SIZE = 0.004
const CORNER_COUNT = 4

/** A point in frame coordinates normalized to [0, 1]. */
export type Point = { x: number; y: number }

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value))
}

/** The four corners of a box: top-left, top-right, bottom-left, bottom-right. */
export function cornersOf(box: Box): [Point, Point, Point, Point] {
  const left = box.cx - box.w / 2
  const right = box.cx + box.w / 2
  const top = box.cy - box.h / 2
  const bottom = box.cy + box.h / 2
  return [
    { x: left, y: top },
    { x: right, y: top },
    { x: left, y: bottom },
    { x: right, y: bottom },
  ]
}

/** Builds a box from two opposite corners, clipped to the frame. Null when it would be degenerate. */
export function boxFromCorners(
  first: Point,
  second: Point,
  robot: Robot | null = null,
): Box | null {
  const left = clamp(Math.min(first.x, second.x), 0, 1)
  const right = clamp(Math.max(first.x, second.x), 0, 1)
  const top = clamp(Math.min(first.y, second.y), 0, 1)
  const bottom = clamp(Math.max(first.y, second.y), 0, 1)
  if (right - left < MIN_BOX_SIZE || bottom - top < MIN_BOX_SIZE) return null
  return { cx: (left + right) / 2, cy: (top + bottom) / 2, w: right - left, h: bottom - top, robot }
}

/** Moves a box by a delta, keeping its size and stopping at the frame edge. */
export function moveBox(box: Box, deltaX: number, deltaY: number): Box {
  return {
    ...box,
    cx: clamp(box.cx + deltaX, box.w / 2, 1 - box.w / 2),
    cy: clamp(box.cy + deltaY, box.h / 2, 1 - box.h / 2),
  }
}

/** Grows or shrinks a box around its centre, never below the minimum size or past the frame. */
export function resizeBox(box: Box, deltaWidth: number, deltaHeight: number): Box {
  const maxWidth = 2 * Math.min(box.cx, 1 - box.cx)
  const maxHeight = 2 * Math.min(box.cy, 1 - box.cy)
  return {
    ...box,
    w: clamp(box.w + deltaWidth, MIN_BOX_SIZE, maxWidth),
    h: clamp(box.h + deltaHeight, MIN_BOX_SIZE, maxHeight),
  }
}

/** Drags one corner of a box to a point, anchoring the opposite corner. */
export function dragCorner(box: Box, cornerIndex: number, point: Point): Box {
  const opposite = cornersOf(box)[CORNER_COUNT - 1 - cornerIndex]
  if (opposite === undefined) return box
  return boxFromCorners(opposite, point, box.robot) ?? box
}

/** Whether a point falls inside a box. */
export function containsPoint(box: Box, point: Point): boolean {
  return Math.abs(point.x - box.cx) <= box.w / 2 && Math.abs(point.y - box.cy) <= box.h / 2
}
