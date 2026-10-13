import { describe, expect, test } from 'bun:test'

import type { ClipDetail } from '@/shared/clip'

import { eventTimes, identityLines } from './exports'

const detail: ClipDetail = {
  dataset: 'train',
  clip: 'match',
  role: 'train',
  source: 'br',
  width: 1000,
  height: 500,
  fps: 30,
  frameCount: 4,
  slotCount: 4,
  boxCount: 4,
  approvedCount: 0,
  rejectedCount: 1,
  bothRobotsCount: 0,
  identifiedCount: 0,
  eventCount: 4,
  roundCount: 0,
  hasVideo: true,
  canExtractFrames: true,
  thumbnailIndex: null,
  stride: 3,
  frames: [
    {
      index: 0,
      hasImage: true,
      review: 'pending',
      boxes: [
        { cx: 0.5, cy: 0.5, w: 0.1, h: 0.2, robot: 'A' },
        { cx: 0.25, cy: 0.5, w: 0.1, h: 0.2, robot: null },
      ],
    },
    {
      index: 2,
      hasImage: true,
      review: 'approved',
      boxes: [{ cx: 0.3, cy: 0.4, w: 0.2, h: 0.1, robot: 'B' }],
    },
    {
      index: 3,
      hasImage: true,
      review: 'rejected',
      boxes: [{ cx: 0.5, cy: 0.5, w: 0.1, h: 0.1, robot: 'A' }],
    },
  ],
  events: [
    { kind: 'round_start', frameIndex: 10, robot: null },
    { kind: 'ring_out', frameIndex: 20, robot: 'A' },
    { kind: 'ring_out', frameIndex: 30, robot: 'B' },
  ],
  rounds: [],
}

describe('identityLines', () => {
  test('writes a track line per named box with pixel geometry', () => {
    expect(identityLines(detail)).toBe('0,0,450.0,200.0,100.0,100.0\n6,1,200.0,175.0,200.0,50.0\n')
  })

  test('leaves out rejected frames and unnamed boxes', () => {
    const lines = identityLines(detail).trim().split('\n')
    expect(lines).toHaveLength(2)
    expect(lines.some((line) => line.startsWith('9,'))).toBe(false)
  })

  test('returns an empty string when no box is named', () => {
    expect(identityLines({ ...detail, frames: [] })).toBe('')
  })
})

describe('eventTimes', () => {
  test('converts slots to milliseconds and groups them by kind', () => {
    expect(eventTimes(detail)).toEqual({ round_start: [1000], ring_out: [2000, 3000] })
  })

  test('falls back to the annotation rate when the clip has no video to read a frame rate from', () => {
    const events = [{ kind: 'first_contact' as const, frameIndex: 5, robot: null }]
    expect(eventTimes({ ...detail, fps: 0, events })).toEqual({ first_contact: [500] })
  })

  test('returns no kinds for a clip without events', () => {
    expect(eventTimes({ ...detail, events: [] })).toEqual({})
  })
})
