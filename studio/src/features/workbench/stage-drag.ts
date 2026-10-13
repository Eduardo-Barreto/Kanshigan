import {
  boxFromCorners,
  containsPoint,
  cornersOf,
  dragCorner,
  moveBox,
  type Point,
} from '@/shared/box-editing'
import type { Box } from '@/shared/clip'

/** A pointer gesture in progress on the frame: moving a box, pulling a corner, or drawing a new box. */
export type Drag =
  | { kind: 'move'; boxIndex: number; origin: Point; original: Box }
  | { kind: 'corner'; boxIndex: number; cornerIndex: number }
  | { kind: 'draw'; origin: Point }

/** What a gesture produces at a given pointer position. */
export type DragResult = { boxes: Box[]; selected: number | null; preview: Box | null }

function cornerUnder(box: Box, point: Point, tolerance: Point): number | null {
  const cornerIndex = cornersOf(box).findIndex(
    (corner) =>
      Math.abs(corner.x - point.x) <= tolerance.x && Math.abs(corner.y - point.y) <= tolerance.y,
  )
  return cornerIndex === -1 ? null : cornerIndex
}

/**
 * Decides what a pointer press starts. A corner of the selected box wins, so a corner that
 * overlaps another box stays grabbable. Otherwise the topmost box under the pointer moves,
 * and a press on empty frame draws.
 */
export function beginDrag(
  boxes: Box[],
  selected: number | null,
  point: Point,
  tolerance: Point,
): Drag {
  const selectedBox = selected === null ? undefined : boxes[selected]
  if (selected !== null && selectedBox !== undefined) {
    const cornerIndex = cornerUnder(selectedBox, point, tolerance)
    if (cornerIndex !== null) return { kind: 'corner', boxIndex: selected, cornerIndex }
  }
  const boxIndex = boxes.findLastIndex((box) => containsPoint(box, point))
  const original = boxes[boxIndex]
  if (original === undefined) return { kind: 'draw', origin: point }
  return { kind: 'move', boxIndex, origin: point, original }
}

function replaceAt(boxes: Box[], boxIndex: number, replacement: Box): Box[] {
  return boxes.map((box, position) => (position === boxIndex ? replacement : box))
}

/** Applies a gesture at the current pointer position. A drawing gesture only yields a preview. */
export function continueDrag(drag: Drag, boxes: Box[], point: Point): DragResult {
  if (drag.kind === 'draw') {
    return { boxes, selected: null, preview: boxFromCorners(drag.origin, point) }
  }
  const current = boxes[drag.boxIndex]
  if (current === undefined) return { boxes, selected: null, preview: null }
  const edited =
    drag.kind === 'move'
      ? moveBox(drag.original, point.x - drag.origin.x, point.y - drag.origin.y)
      : dragCorner(current, drag.cornerIndex, point)
  return { boxes: replaceAt(boxes, drag.boxIndex, edited), selected: drag.boxIndex, preview: null }
}

/** Finishes a gesture: a drawn box joins the frame and becomes the selection. */
export function endDrag(drag: Drag, boxes: Box[], point: Point): DragResult {
  const result = continueDrag(drag, boxes, point)
  if (result.preview === null) return result
  return { boxes: [...boxes, result.preview], selected: boxes.length, preview: null }
}
