import { describe, expect, test } from 'bun:test'

import type { Box, ClipDetail, Frame, Robot, SaveFrame, SaveTimeline } from '@/shared/clip'

import { WorkbenchStore, currentBoxes, currentFrame } from './workbench-store'

const boxAt = (cx: number, robot: Robot | null = null): Box => ({
  cx,
  cy: 0.5,
  w: 0.1,
  h: 0.1,
  robot,
})

const frameWith = (index: number, boxes: Box[], overrides: Partial<Frame> = {}): Frame => ({
  index,
  hasImage: true,
  boxes,
  review: 'pending',
  ...overrides,
})

function clipWith(frames: Frame[]): ClipDetail {
  return {
    dataset: 'gold_candidates',
    clip: 'round',
    role: 'gold_candidate',
    source: 'br',
    width: 1000,
    height: 500,
    fps: 60,
    stride: 6,
    frameCount: frames.length,
    slotCount: frames.length,
    boxCount: 0,
    approvedCount: 0,
    rejectedCount: 0,
    bothRobotsCount: 0,
    identifiedCount: 0,
    eventCount: 0,
    roundCount: 0,
    hasVideo: true,
    canExtractFrames: true,
    thumbnailIndex: 0,
    frames,
    events: [],
    rounds: [],
  }
}

function workbenchOver(frames: Frame[]) {
  const savedFrames: (SaveFrame & { index: number })[][] = []
  const savedTimelines: SaveTimeline[] = []
  const store = new WorkbenchStore(clipWith(frames), {
    saveFrames: (saves) => {
      savedFrames.push(saves)
      return Promise.resolve()
    },
    saveTimeline: (timeline) => {
      savedTimelines.push(timeline)
      return Promise.resolve()
    },
    preloadFrame: () => undefined,
    onDetailChange: () => undefined,
  })
  return { store, savedFrames, savedTimelines }
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 0))

describe('editing a frame', () => {
  test('keeps edits in a draft until the cursor leaves the frame', async () => {
    const { store, savedFrames } = workbenchOver([frameWith(0, [boxAt(0.2)]), frameWith(1, [])])
    store.edit([boxAt(0.4)])
    expect(savedFrames).toHaveLength(0)

    store.step(1)
    await settle()

    expect(savedFrames).toEqual([[{ index: 0, boxes: [boxAt(0.4)], review: 'pending' }]])
    expect(store.getSnapshot().detail.frames[0]?.boxes).toEqual([boxAt(0.4)])
  })

  test('does not save a frame nobody touched', async () => {
    const { store, savedFrames } = workbenchOver([frameWith(0, [boxAt(0.2)]), frameWith(1, [])])
    store.step(1)
    await settle()
    expect(savedFrames).toHaveLength(0)
  })

  test('selects the box that remains after deleting the selected one', () => {
    const { store } = workbenchOver([frameWith(0, [boxAt(0.2), boxAt(0.8)])])
    store.deleteSelected()
    expect(currentBoxes(store.getSnapshot())).toEqual([boxAt(0.2)])
    expect(store.getSnapshot().selected).toBe(0)
  })

  test('starts on the first frame that has an image', () => {
    const { store } = workbenchOver([
      frameWith(0, [], { hasImage: false }),
      frameWith(1, [boxAt(0.5)]),
    ])
    expect(currentFrame(store.getSnapshot()).index).toBe(1)
  })
})

describe('reviewing', () => {
  test('saves the status with the draft boxes and moves to the next frame', async () => {
    const { store, savedFrames } = workbenchOver([frameWith(0, [boxAt(0.2)]), frameWith(1, [])])
    store.edit([boxAt(0.3)])
    store.review('approved')
    await settle()

    expect(savedFrames).toEqual([[{ index: 0, boxes: [boxAt(0.3)], review: 'approved' }]])
    expect(store.getSnapshot().position).toBe(1)
  })

  test('keeps a slot the annotator filled in, marking it as having an image', () => {
    const { store } = workbenchOver([frameWith(0, [], { hasImage: false })])
    store.edit([boxAt(0.5)])
    store.flush()
    expect(store.getSnapshot().detail.frames[0]?.hasImage).toBe(true)
  })

  test('jumps to the next frame that is not showing two robots', () => {
    const pair = [boxAt(0.2), boxAt(0.8)]
    const { store } = workbenchOver([
      frameWith(0, pair),
      frameWith(1, pair),
      frameWith(2, [boxAt(0.5)]),
    ])
    store.nextAnomaly()
    expect(store.getSnapshot().position).toBe(2)
  })
})

describe('approving in bulk', () => {
  test('approves every pending labeled frame in one save and leaves the rest alone', async () => {
    const { store, savedFrames } = workbenchOver([
      frameWith(0, [boxAt(0.2)]),
      frameWith(1, [], { review: 'rejected' }),
      frameWith(2, [], { hasImage: false }),
      frameWith(3, [boxAt(0.4)]),
    ])
    expect(store.approvePending()).toBe(2)
    await settle()

    expect(savedFrames).toHaveLength(1)
    expect(savedFrames[0]?.map((save) => save.index)).toEqual([0, 3])
    expect(store.getSnapshot().detail.frames.map((frame) => frame.review)).toEqual([
      'approved',
      'rejected',
      'pending',
      'approved',
    ])
  })
})

describe('identity', () => {
  test('proposes the previous frame identities when entering an unnamed frame', () => {
    const { store } = workbenchOver([
      frameWith(0, [boxAt(0.2, 'A'), boxAt(0.8, 'B')]),
      frameWith(1, [boxAt(0.78), boxAt(0.22)]),
    ])
    store.step(1)
    expect(currentBoxes(store.getSnapshot()).map((box) => box.robot)).toEqual(['B', 'A'])
  })

  test('does not propose when propagation is switched off', () => {
    const { store } = workbenchOver([
      frameWith(0, [boxAt(0.2, 'A'), boxAt(0.8, 'B')]),
      frameWith(1, [boxAt(0.78), boxAt(0.22)]),
    ])
    store.setOption('propagate', false)
    store.step(1)
    expect(store.getSnapshot().draft).toBeNull()
  })

  test('carries identity to the end of the clip across a frame without boxes', async () => {
    const { store, savedFrames } = workbenchOver([
      frameWith(0, [boxAt(0.2, 'A'), boxAt(0.8, 'B')]),
      frameWith(1, []),
      frameWith(2, [boxAt(0.75), boxAt(0.25)]),
    ])
    store.propagateForward()
    await settle()

    expect(savedFrames).toHaveLength(1)
    expect(savedFrames[0]?.map((save) => save.index)).toEqual([2])
    expect(store.getSnapshot().detail.frames[2]?.boxes.map((box) => box.robot)).toEqual(['B', 'A'])
  })
})

describe('timeline', () => {
  test('removes an event when the same kind is marked twice on a frame', async () => {
    const { store, savedTimelines } = workbenchOver([frameWith(0, [])])
    store.toggleEvent('round_start')
    store.toggleEvent('round_start')
    await settle()

    expect(savedTimelines.map((timeline) => timeline.events.length)).toEqual([1, 0])
  })

  test('closes a round from the two marks, whichever came first', async () => {
    const { store, savedTimelines } = workbenchOver([
      frameWith(0, []),
      frameWith(1, []),
      frameWith(2, []),
    ])
    store.step(2)
    store.markRound()
    store.step(-2)
    store.markRound()
    await settle()

    expect(savedTimelines.at(-1)?.rounds).toEqual([{ start: 0, end: 2 }])
  })
})

describe('proposed identity', () => {
  const named = [boxAt(0.2, 'A'), boxAt(0.8, 'B')]
  const unnamed = [boxAt(0.22), boxAt(0.78)]

  test('is saved when the annotator steps on to the next frame', async () => {
    const { store, savedFrames } = workbenchOver([
      frameWith(0, named),
      frameWith(1, unnamed),
      frameWith(2, unnamed),
    ])
    store.step(1)
    store.step(1)
    await settle()

    expect(savedFrames.map((saves) => saves[0]?.index)).toEqual([1])
  })

  test('is dropped, not saved, when the annotator only browses back', async () => {
    const { store, savedFrames } = workbenchOver([frameWith(0, named), frameWith(1, unnamed)])
    store.step(1)
    store.step(-1)
    await settle()

    expect(savedFrames).toHaveLength(0)
    expect(store.getSnapshot().detail.frames[1]?.boxes).toEqual(unnamed)
  })

  test('is dropped when the workbench closes', async () => {
    const { store, savedFrames } = workbenchOver([frameWith(0, named), frameWith(1, unnamed)])
    store.step(1)
    store.flush()
    await settle()

    expect(savedFrames).toHaveLength(0)
  })

  test('repairs frames that carried a swapped identity when propagated again', async () => {
    const { store } = workbenchOver([
      frameWith(0, named),
      frameWith(1, [boxAt(0.22, 'B'), boxAt(0.78, 'A')]),
    ])
    store.propagateForward()
    await settle()

    expect(store.getSnapshot().detail.frames[1]?.boxes.map((box) => box.robot)).toEqual(['A', 'B'])
  })
})

describe('reviewing a gap', () => {
  test('moves past an unlabeled slot without turning it into an approved empty frame', async () => {
    const { store, savedFrames } = workbenchOver([
      frameWith(0, [boxAt(0.5)]),
      frameWith(1, [], { hasImage: false }),
      frameWith(2, []),
    ])
    store.step(1)
    store.review('approved')
    await settle()

    expect(savedFrames).toHaveLength(0)
    expect(store.getSnapshot().position).toBe(2)
  })
})

describe('a save that fails', () => {
  test('is sent again before newer saves and keeps the error until it lands', async () => {
    const attempts: number[] = []
    let isOffline = true
    const store = new WorkbenchStore(
      clipWith([frameWith(0, []), frameWith(1, []), frameWith(2, [])]),
      {
        saveFrames: (saves) => {
          attempts.push(saves[0]?.index ?? -1)
          return isOffline ? Promise.reject(new Error('sem rede')) : Promise.resolve()
        },
        saveTimeline: () => Promise.resolve(),
        preloadFrame: () => undefined,
        onDetailChange: () => undefined,
      },
    )
    store.edit([boxAt(0.5)])
    store.step(1)
    await settle()
    expect(store.getSnapshot()).toMatchObject({ saveError: 'sem rede', failedSaves: 1 })

    isOffline = false
    store.edit([boxAt(0.6)])
    store.step(1)
    await settle()

    expect(attempts).toEqual([0, 0, 1])
    expect(store.getSnapshot()).toMatchObject({ saveError: null, failedSaves: 0 })
  })
})

describe('saving', () => {
  test('reports the error of a failed save and keeps the local edit', async () => {
    const store = new WorkbenchStore(clipWith([frameWith(0, []), frameWith(1, [])]), {
      saveFrames: () => Promise.reject(new Error('disco cheio')),
      saveTimeline: () => Promise.resolve(),
      preloadFrame: () => undefined,
      onDetailChange: () => undefined,
    })
    store.edit([boxAt(0.5)])
    store.step(1)
    await settle()

    expect(store.getSnapshot().saveError).toBe('disco cheio')
    expect(store.getSnapshot().detail.frames[0]?.boxes).toEqual([boxAt(0.5)])
  })
})
