import type { LabelBox } from './labels'

/** Intersection over union of two normalized centre-size boxes. */
export function intersectionOverUnion(first: LabelBox, second: LabelBox): number {
  const overlapWidth =
    Math.min(first.cx + first.w / 2, second.cx + second.w / 2) -
    Math.max(first.cx - first.w / 2, second.cx - second.w / 2)
  const overlapHeight =
    Math.min(first.cy + first.h / 2, second.cy + second.h / 2) -
    Math.max(first.cy - first.h / 2, second.cy - second.h / 2)
  if (overlapWidth <= 0 || overlapHeight <= 0) return 0
  const intersection = overlapWidth * overlapHeight
  return intersection / (first.w * first.h + second.w * second.h - intersection)
}
