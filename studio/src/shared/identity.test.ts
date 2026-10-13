import { describe, expect, test } from 'bun:test'

import type { Box, Robot } from './clip'
import { assignRobot, isIdentified, propagateIdentity, swapRobots } from './identity'

const boxAt = (cx: number, robot: Robot | null = null): Box => ({
  cx,
  cy: 0.5,
  w: 0.1,
  h: 0.1,
  robot,
})
const robotsOf = (boxes: Box[]) => boxes.map((box) => box.robot)

describe('assignRobot', () => {
  test('gives the other box of a pair the opposite identity', () => {
    expect(robotsOf(assignRobot([boxAt(0.2), boxAt(0.8)], 1, 'A'))).toEqual(['B', 'A'])
  })

  test('takes the identity away from a box that already held it when there are three boxes', () => {
    const boxes = [boxAt(0.2, 'A'), boxAt(0.5), boxAt(0.8, 'B')]
    expect(robotsOf(assignRobot(boxes, 1, 'A'))).toEqual([null, 'A', 'B'])
  })

  test('names a lone box without inventing a second robot', () => {
    expect(robotsOf(assignRobot([boxAt(0.5)], 0, 'B'))).toEqual(['B'])
  })
})

describe('swapRobots', () => {
  test('exchanges A and B and leaves unnamed boxes alone', () => {
    expect(robotsOf(swapRobots([boxAt(0.2, 'A'), boxAt(0.5), boxAt(0.8, 'B')]))).toEqual([
      'B',
      null,
      'A',
    ])
  })
})

describe('propagateIdentity', () => {
  test('follows each robot to the nearest box in the next frame', () => {
    const previous = [boxAt(0.2, 'A'), boxAt(0.8, 'B')]
    const current = [boxAt(0.75), boxAt(0.25)]
    expect(robotsOf(propagateIdentity(previous, current))).toEqual(['B', 'A'])
  })

  test('hands each identity out once when both robots are near the same box', () => {
    const previous = [boxAt(0.5, 'A'), boxAt(0.52, 'B')]
    const current = [boxAt(0.5), boxAt(0.9)]
    expect(robotsOf(propagateIdentity(previous, current))).toEqual(['A', 'B'])
  })

  test('keeps an identity a human already set on the current frame', () => {
    const previous = [boxAt(0.2, 'A'), boxAt(0.8, 'B')]
    const current = [boxAt(0.2, 'B'), boxAt(0.8)]
    expect(robotsOf(propagateIdentity(previous, current))).toEqual(['B', 'A'])
  })

  test('leaves the frame unnamed when the previous frame has no identity', () => {
    expect(robotsOf(propagateIdentity([boxAt(0.2)], [boxAt(0.2)]))).toEqual([null])
  })

  test('names only one box when a robot left the frame', () => {
    const previous = [boxAt(0.2, 'A'), boxAt(0.8, 'B')]
    expect(robotsOf(propagateIdentity(previous, [boxAt(0.78)]))).toEqual(['B'])
  })
})

describe('isIdentified', () => {
  test('requires every box to be named and at least one box', () => {
    expect(isIdentified([])).toBe(false)
    expect(isIdentified([boxAt(0.2, 'A'), boxAt(0.8)])).toBe(false)
    expect(isIdentified([boxAt(0.2, 'A'), boxAt(0.8, 'B')])).toBe(true)
  })
})
