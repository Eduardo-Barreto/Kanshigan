import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import type { ClipDetail } from '@/shared/clip'

import { moveBlockers, moveClip } from './clip-move'
import { loadClip, toDetail, toSummary } from './clip-read'
import { resolveFrameImage, saveFrames, saveTimeline } from './clip-write'
import { loadConfig, type StudioConfig } from './config'
import { readLibrary } from './library'
import { Problem } from './problem'

const IMAGE_WIDTH = 64
const IMAGE_HEIGHT = 48
const ROBOT_BOX = '0 0.500000 0.500000 0.100000 0.200000\n'
const OTHER_BOX = '0 0.250000 0.250000 0.100000 0.100000\n'

let dataRoot = ''
let config: StudioConfig

beforeEach(async () => {
  dataRoot = await mkdtemp(join(tmpdir(), 'studio-store-'))
  config = loadConfig({ STUDIO_DATA_ROOT: dataRoot })
})

afterEach(async () => {
  await rm(dataRoot, { recursive: true, force: true })
})

function jpegBytes(width: number, height: number): Uint8Array {
  const appSegment = [0xff, 0xe0, 0x00, 0x10, ...Array.from({ length: 14 }, () => 0)]
  return Uint8Array.from([
    0xff,
    0xd8,
    ...appSegment,
    0xff,
    0xc0,
    0x00,
    0x0b,
    0x08,
    height >> 8,
    height & 0xff,
    width >> 8,
    width & 0xff,
    0x01,
    0x01,
    0x11,
    0x00,
  ])
}

const datasetPath = (dataset: string, ...segments: string[]) =>
  join(config.paths.annotations, dataset, ...segments)

const stemOf = (clip: string, index: number) => `${clip}_${String(index).padStart(4, '0')}`

async function writeFrame(dataset: string, clip: string, index: number, labels: string) {
  const stem = stemOf(clip, index)
  await Bun.write(
    datasetPath(dataset, 'images', `${stem}.jpg`),
    jpegBytes(IMAGE_WIDTH, IMAGE_HEIGHT),
  )
  await Bun.write(datasetPath(dataset, 'labels', `${stem}.txt`), labels)
}

const writeSidecarJson = (dataset: string, clip: string, sidecar: object) =>
  Bun.write(datasetPath(dataset, 'studio', `${clip}.json`), JSON.stringify(sidecar))

const exists = (path: string) => Bun.file(path).exists()

async function rejectionOf(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise
  } catch (error) {
    return error
  }
  return undefined
}

async function seedThreeFrames(dataset: string, clip: string) {
  await writeFrame(dataset, clip, 0, ROBOT_BOX + OTHER_BOX)
  await writeFrame(dataset, clip, 1, ROBOT_BOX)
  await writeFrame(dataset, clip, 2, '')
}

const reviewsOf = (detail: ClipDetail) => detail.frames.map((frame) => frame.review)

describe('loadClip, toSummary and toDetail', () => {
  test('reads frames with their boxes', async () => {
    await seedThreeFrames('train', 'match')
    const detail = toDetail(await loadClip(config, { dataset: 'train', clip: 'match' }))
    expect(detail.frames.map((frame) => frame.index)).toEqual([0, 1, 2])
    expect(detail.frames.map((frame) => frame.boxes.length)).toEqual([2, 1, 0])
    expect(detail.frames[1]?.boxes[0]).toEqual({ cx: 0.5, cy: 0.5, w: 0.1, h: 0.2, robot: null })
  })

  test('summarizes counts and image size', async () => {
    await seedThreeFrames('train', 'match')
    await writeSidecarJson('train', 'match', {
      version: 1,
      review: { '0': 'approved', '2': 'rejected' },
      identity: { '1': ['A'] },
      events: [{ kind: 'round_start', frameIndex: 0, robot: null }],
      rounds: [{ start: 0, end: 2 }],
    })
    const summary = toSummary(await loadClip(config, { dataset: 'train', clip: 'match' }))
    expect(summary).toMatchObject({
      dataset: 'train',
      clip: 'match',
      role: 'train',
      width: IMAGE_WIDTH,
      height: IMAGE_HEIGHT,
      frameCount: 3,
      slotCount: 3,
      boxCount: 3,
      approvedCount: 1,
      rejectedCount: 1,
      bothRobotsCount: 1,
      identifiedCount: 1,
      eventCount: 1,
      roundCount: 1,
      hasVideo: false,
      canExtractFrames: false,
    })
  })

  test('rejects a missing clip with a 404 problem', async () => {
    const error = await rejectionOf(loadClip(config, { dataset: 'train', clip: 'ghost' }))
    expect(error).toBeInstanceOf(Problem)
    expect(error).toMatchObject({ status: 404 })
  })

  test('defaults frames of a gold dataset to approved', async () => {
    await seedThreeFrames('gold_crop', 'match')
    const detail = toDetail(await loadClip(config, { dataset: 'gold_crop', clip: 'match' }))
    expect(reviewsOf(detail)).toEqual(['approved', 'approved', 'approved'])
  })

  test('defaults frames of other datasets to pending', async () => {
    await seedThreeFrames('train', 'match')
    const detail = toDetail(await loadClip(config, { dataset: 'train', clip: 'match' }))
    expect(reviewsOf(detail)).toEqual(['pending', 'pending', 'pending'])
  })

  test('drops identity when its length no longer matches the label lines', async () => {
    await writeFrame('train', 'match', 0, ROBOT_BOX)
    await writeFrame('train', 'match', 1, ROBOT_BOX + OTHER_BOX)
    await writeSidecarJson('train', 'match', {
      version: 1,
      review: {},
      identity: { '0': ['A', 'B'], '1': ['A', 'B'] },
      events: [],
      rounds: [],
    })
    const detail = toDetail(await loadClip(config, { dataset: 'train', clip: 'match' }))
    expect(detail.frames[0]?.boxes.map((box) => box.robot)).toEqual([null])
    expect(detail.frames[1]?.boxes.map((box) => box.robot)).toEqual(['A', 'B'])
  })
})

describe('legacy import', () => {
  test('reads review.json and round marks when the clip has no sidecar', async () => {
    await seedThreeFrames('train', 'match')
    await Bun.write(
      datasetPath('train', 'review.json'),
      JSON.stringify({ match_0000: 'approved', match_0001: 'rejected', match_0002: 'unreviewed' }),
    )
    await Bun.write(
      join(config.paths.annotations, 'round_marks', 'match.json'),
      JSON.stringify({ stride: 6, rounds: [[0, 2]] }),
    )
    const detail = toDetail(await loadClip(config, { dataset: 'train', clip: 'match' }))
    expect(reviewsOf(detail)).toEqual(['approved', 'rejected', 'pending'])
    expect(detail.rounds).toEqual([{ start: 0, end: 2 }])
  })

  test('converts round marks of the first format from native frames to slots and drops the ones past the clip', async () => {
    await seedThreeFrames('train', 'match')
    await Bun.write(
      join(config.paths.annotations, 'round_marks', 'match.json'),
      JSON.stringify({
        n_frames: 18,
        rounds: [
          [0, 12],
          [60, 600],
        ],
      }),
    )
    const detail = toDetail(await loadClip(config, { dataset: 'train', clip: 'match' }))
    expect(detail.rounds).toEqual([{ start: 0, end: 2 }])
  })

  test('imports the gold BR identity file for its own clip even after another clip joins the dataset', async () => {
    await seedThreeFrames('gold_crop', 'gold_zb01')
    await seedThreeFrames('gold_crop', 'newcomer')
    await Bun.write(datasetPath('gold_crop', 'identity.txt'), '0,0,28.8,19.2,6.4,9.6\n')

    const original = toDetail(await loadClip(config, { dataset: 'gold_crop', clip: 'gold_zb01' }))
    const newcomer = toDetail(await loadClip(config, { dataset: 'gold_crop', clip: 'newcomer' }))

    expect(original.frames[0]?.boxes.map((box) => box.robot)).toContain('A')
    expect(newcomer.frames[0]?.boxes.every((box) => box.robot === null)).toBe(true)
  })

  test('keeps a moved gold clip approved by writing out the review its old dataset implied', async () => {
    await seedThreeFrames('gold_candidates', 'match')
    await seedThreeFrames('gold_crop', 'gold_demo')
    const gold = await loadClip(config, { dataset: 'gold_crop', clip: 'gold_demo' })
    await moveClip(config, gold, 'gold_candidates')

    const moved = toDetail(
      await loadClip(config, { dataset: 'gold_candidates', clip: 'gold_demo' }),
    )
    expect(reviewsOf(moved)).toEqual(['approved', 'approved', 'approved'])
  })

  test('ignores legacy files once the clip has its own sidecar', async () => {
    await seedThreeFrames('train', 'match')
    await Bun.write(datasetPath('train', 'review.json'), JSON.stringify({ match_0000: 'approved' }))
    await writeSidecarJson('train', 'match', {
      version: 1,
      review: {},
      identity: {},
      events: [],
      rounds: [],
    })
    const detail = toDetail(await loadClip(config, { dataset: 'train', clip: 'match' }))
    expect(reviewsOf(detail)).toEqual(['pending', 'pending', 'pending'])
  })
})

describe('saveFrames', () => {
  test('rewrites the label file and records review and identity in the sidecar', async () => {
    await seedThreeFrames('train', 'match')
    const clip = await loadClip(config, { dataset: 'train', clip: 'match' })
    await saveFrames(clip, [
      {
        index: 1,
        review: 'approved',
        boxes: [
          { cx: 0.4, cy: 0.4, w: 0.2, h: 0.2, robot: 'B' },
          { cx: 0.7, cy: 0.7, w: 0.1, h: 0.1, robot: 'A' },
        ],
      },
    ])
    expect(await Bun.file(datasetPath('train', 'labels', 'match_0001.txt')).text()).toBe(
      '0 0.400000 0.400000 0.200000 0.200000\n0 0.700000 0.700000 0.100000 0.100000\n',
    )
    const reread = toDetail(await loadClip(config, { dataset: 'train', clip: 'match' }))
    expect(reread.frames[1]?.review).toBe('approved')
    expect(reread.frames[1]?.boxes.map((box) => box.robot)).toEqual(['B', 'A'])
    expect(reread.frames[0]?.review).toBe('pending')
  })

  test('moves a rejected frame under rejected/ and back when approved again', async () => {
    await seedThreeFrames('train', 'match')
    const activeImage = datasetPath('train', 'images', 'match_0001.jpg')
    const rejectedImage = datasetPath('train', 'rejected', 'images', 'match_0001.jpg')
    const rejectedLabel = datasetPath('train', 'rejected', 'labels', 'match_0001.txt')
    const boxes = [{ cx: 0.5, cy: 0.5, w: 0.1, h: 0.2, robot: null }]

    await saveFrames(await loadClip(config, { dataset: 'train', clip: 'match' }), [
      { index: 1, review: 'rejected', boxes },
    ])
    expect(await exists(activeImage)).toBe(false)
    expect(await exists(datasetPath('train', 'labels', 'match_0001.txt'))).toBe(false)
    expect(await exists(rejectedImage)).toBe(true)
    expect(await exists(rejectedLabel)).toBe(true)
    const rejected = toDetail(await loadClip(config, { dataset: 'train', clip: 'match' }))
    expect(rejected.frames[1]).toMatchObject({ review: 'rejected', hasImage: true })

    await saveFrames(await loadClip(config, { dataset: 'train', clip: 'match' }), [
      { index: 1, review: 'approved', boxes },
    ])
    expect(await exists(activeImage)).toBe(true)
    expect(await exists(datasetPath('train', 'labels', 'match_0001.txt'))).toBe(true)
    expect(await exists(rejectedImage)).toBe(false)
    expect(await exists(rejectedLabel)).toBe(false)
  })
})

describe('saveTimeline', () => {
  test('sorts events and rounds by position', async () => {
    await seedThreeFrames('train', 'match')
    const clip = await loadClip(config, { dataset: 'train', clip: 'match' })
    await saveTimeline(clip, {
      events: [
        { kind: 'ring_out', frameIndex: 2, robot: 'A' },
        { kind: 'round_start', frameIndex: 0, robot: null },
      ],
      rounds: [
        { start: 2, end: 2 },
        { start: 0, end: 1 },
      ],
    })
    const detail = toDetail(await loadClip(config, { dataset: 'train', clip: 'match' }))
    expect(detail.events.map((event) => event.frameIndex)).toEqual([0, 2])
    expect(detail.rounds.map((round) => round.start)).toEqual([0, 2])
  })

  test('refuses an event past the last slot', async () => {
    await seedThreeFrames('train', 'match')
    const clip = await loadClip(config, { dataset: 'train', clip: 'match' })
    const error = await rejectionOf(
      saveTimeline(clip, {
        events: [{ kind: 'ring_out', frameIndex: 3, robot: null }],
        rounds: [],
      }),
    )
    expect(error).toMatchObject({ status: 422 })
  })

  test('refuses a round that ends past the last slot', async () => {
    await seedThreeFrames('train', 'match')
    const clip = await loadClip(config, { dataset: 'train', clip: 'match' })
    const error = await rejectionOf(
      saveTimeline(clip, { events: [], rounds: [{ start: 0, end: 3 }] }),
    )
    expect(error).toMatchObject({ status: 422 })
  })
})

describe('resolveFrameImage', () => {
  test('returns the dataset image of a frame that has one', async () => {
    await seedThreeFrames('train', 'match')
    const clip = await loadClip(config, { dataset: 'train', clip: 'match' })
    expect(await resolveFrameImage(clip, 1)).toBe(datasetPath('train', 'images', 'match_0001.jpg'))
  })

  test('rejects with a 404 problem when the slot has no image and cannot be extracted', async () => {
    await seedThreeFrames('train', 'match')
    const clip = await loadClip(config, { dataset: 'train', clip: 'match' })
    expect(await rejectionOf(resolveFrameImage(clip, 7))).toMatchObject({ status: 404 })
  })
})

describe('moveClip', () => {
  test('moves images, labels and sidecar to the target dataset', async () => {
    await seedThreeFrames('train', 'match')
    await writeFrame('scratch', 'other', 0, ROBOT_BOX)
    await saveFrames(await loadClip(config, { dataset: 'train', clip: 'match' }), [
      {
        index: 0,
        review: 'approved',
        boxes: [{ cx: 0.5, cy: 0.5, w: 0.1, h: 0.2, robot: 'A' }],
      },
    ])

    const destination = await moveClip(
      config,
      await loadClip(config, { dataset: 'train', clip: 'match' }),
      'scratch',
    )

    expect(destination).toEqual({ dataset: 'scratch', clip: 'match' })
    expect(await exists(datasetPath('scratch', 'images', 'match_0002.jpg'))).toBe(true)
    expect(await exists(datasetPath('scratch', 'labels', 'match_0000.txt'))).toBe(true)
    expect(await exists(datasetPath('scratch', 'studio', 'match.json'))).toBe(true)
    expect(await exists(datasetPath('train', 'images', 'match_0000.jpg'))).toBe(false)
    expect(await exists(datasetPath('train', 'studio', 'match.json'))).toBe(false)
    const moved = toDetail(await loadClip(config, destination))
    expect(moved.frames[0]).toMatchObject({ review: 'approved' })
    expect(moved.frames[0]?.boxes[0]?.robot).toBe('A')

    const library = await readLibrary(config)
    const clipsByDataset = new Map(
      library.datasets.map((dataset) => [dataset.dataset, dataset.clips.map((clip) => clip.clip)]),
    )
    expect(clipsByDataset.get('scratch')).toEqual(['match', 'other'])
    expect(clipsByDataset.has('train')).toBe(false)
  })

  test('rejects with a problem when the target already has a clip of that name', async () => {
    await seedThreeFrames('train', 'match')
    await writeFrame('scratch', 'match', 0, ROBOT_BOX)
    const error = await rejectionOf(
      moveClip(config, await loadClip(config, { dataset: 'train', clip: 'match' }), 'scratch'),
    )
    expect(error).toBeInstanceOf(Problem)
    expect(error).toMatchObject({ status: 409 })
    expect(await exists(datasetPath('train', 'images', 'match_0000.jpg'))).toBe(true)
  })

  test('refuses a move that has blockers', async () => {
    await seedThreeFrames('scratch', 'match')
    const error = await rejectionOf(
      moveClip(config, await loadClip(config, { dataset: 'scratch', clip: 'match' }), 'train'),
    )
    expect(error).toMatchObject({ status: 422 })
  })
})

describe('moveBlockers', () => {
  test('reports pending frames when the target is train', async () => {
    await seedThreeFrames('scratch', 'match')
    const clip = await loadClip(config, { dataset: 'scratch', clip: 'match' })
    const blockers = await moveBlockers(clip, 'train')
    expect(blockers).toHaveLength(1)
    expect(blockers[0]).toContain('3 quadros')
  })

  test('reports nothing for train when every frame is reviewed', async () => {
    await seedThreeFrames('scratch', 'match')
    await writeSidecarJson('scratch', 'match', {
      version: 1,
      review: { '0': 'approved', '1': 'approved', '2': 'rejected' },
      identity: {},
      events: [],
      rounds: [],
    })
    const clip = await loadClip(config, { dataset: 'scratch', clip: 'match' })
    expect(await moveBlockers(clip, 'train')).toEqual([])
  })

  test('does not require review for a non-frozen target', async () => {
    await seedThreeFrames('train', 'match')
    const clip = await loadClip(config, { dataset: 'train', clip: 'match' })
    expect(await moveBlockers(clip, 'scratch')).toEqual([])
  })

  test('reports the full-frame and wrong-source blockers for a gold target', async () => {
    await seedThreeFrames('train', 'match')
    await writeSidecarJson('train', 'match', {
      version: 1,
      review: { '0': 'approved', '1': 'approved', '2': 'approved' },
      identity: {},
      events: [],
      rounds: [],
    })
    const clip = await loadClip(config, { dataset: 'train', clip: 'match' })
    const blockers = await moveBlockers(clip, 'gold_crop')
    expect(blockers.some((blocker) => blocker.includes('quadro inteiro'))).toBe(true)
    expect(blockers.some((blocker) => blocker.includes('o gold dos vídeos'))).toBe(true)
  })
})

describe('readLibrary', () => {
  test('lists datasets gold first and skips the round_marks directory', async () => {
    await writeFrame('scratch', 'c', 0, ROBOT_BOX)
    await writeFrame('train', 'b', 0, ROBOT_BOX)
    await writeFrame('gold_crop', 'a', 0, ROBOT_BOX)
    await writeFrame('round_marks', 'z', 0, ROBOT_BOX)
    const library = await readLibrary(config)
    expect(library.datasets.map((dataset) => dataset.dataset)).toEqual([
      'gold_crop',
      'train',
      'scratch',
    ])
    expect(library.datasets[0]?.clips.map((clip) => clip.clip)).toEqual(['a'])
  })

  test('returns no datasets when the annotations directory is missing', async () => {
    expect(await readLibrary(config)).toEqual({ datasets: [] })
  })
})
