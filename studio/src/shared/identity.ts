import type { Box, Robot } from './clip'

const OTHER_ROBOT: Record<Robot, Robot> = { A: 'B', B: 'A' }
const ROBOTS_PER_ROUND = 2

/** Whether every box of a frame carries a robot identity. An empty frame is not identified. */
export function isIdentified(boxes: Box[]): boolean {
  return boxes.length > 0 && boxes.every((box) => box.robot !== null)
}

/**
 * Names one box. No two boxes keep the same robot: with exactly two boxes the other one
 * takes the opposite identity, otherwise a box that held this identity loses it.
 */
export function assignRobot(boxes: Box[], boxIndex: number, robot: Robot): Box[] {
  const isPair = boxes.length === ROBOTS_PER_ROUND
  return boxes.map((box, position) => {
    if (position === boxIndex) return { ...box, robot }
    if (isPair) return { ...box, robot: OTHER_ROBOT[robot] }
    return box.robot === robot ? { ...box, robot: null } : box
  })
}

/** Exchanges A and B across a frame's boxes. */
export function swapRobots(boxes: Box[]): Box[] {
  return boxes.map((box) => (box.robot ? { ...box, robot: OTHER_ROBOT[box.robot] } : box))
}

function squaredDistance(first: Box, second: Box): number {
  return (first.cx - second.cx) ** 2 + (first.cy - second.cy) ** 2
}

/**
 * Carries robot identities from one frame to the next by nearest centre, the same
 * continuity assumption `identity_gold.py` uses. Each identity is handed out once, to the
 * closest unnamed box. Boxes that already have an identity keep it, and a crossing can
 * still swap the pair, so the result is a proposal for a human to confirm.
 */
export function propagateIdentity(previous: Box[], current: Box[]): Box[] {
  const taken = new Set(current.map((box) => box.robot).filter((robot) => robot !== null))
  const pairs = previous
    .filter((source) => source.robot !== null && !taken.has(source.robot))
    .flatMap((source) =>
      current.map((target, targetIndex) => ({
        robot: source.robot,
        targetIndex,
        distance: squaredDistance(source, target),
      })),
    )
    .filter((pair) => current[pair.targetIndex]?.robot === null)
    .toSorted((first, second) => first.distance - second.distance)

  const assigned = new Map<number, Robot>()
  for (const pair of pairs) {
    if (pair.robot === null || taken.has(pair.robot) || assigned.has(pair.targetIndex)) continue
    assigned.set(pair.targetIndex, pair.robot)
    taken.add(pair.robot)
  }
  return current.map((box, position) => ({ ...box, robot: assigned.get(position) ?? box.robot }))
}
