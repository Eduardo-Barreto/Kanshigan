import { join } from 'node:path'

import { z } from 'zod'

import type { Robot } from '@/shared/clip'

import type { ClipLocation, ClipPaths } from './clip-layout'
import { parseFrameFile } from './clip-layout'
import type { StudioConfig } from './config'
import { readJsonFile, readTextFile } from './files'
import { intersectionOverUnion } from './geometry'
import type { ImageSize } from './jpeg-size'
import type { LabelBox } from './labels'
import { emptySidecar, type Sidecar } from './sidecar'

const IDENTITY_MATCH_IOU = 0.5
const IDENTITY_FIELDS = 6
const ROBOT_BY_TRACK_ID: Record<number, Robot> = { 0: 'A', 1: 'B' }
// `identity.txt` names no clip: `identity_gold.py` wrote it for the one gold BR round. Tying
// it to that clip by name keeps the import working when other clips join the dataset.
const IDENTITY_FILE_CLIP: Record<string, string> = { gold_crop: 'gold_zb01' }

const legacyReviewSchema = z.record(z.string(), z.enum(['approved', 'rejected', 'unreviewed']))
const legacyRoundMarksSchema = z.object({
  stride: z.number().int().positive().optional(),
  rounds: z.array(z.tuple([z.number().int(), z.number().int()])),
})

/** What the importer needs to know about the clip it is importing annotations for. */
export type LegacyContext = {
  location: ClipLocation
  paths: ClipPaths
  labels: Map<number, LabelBox[]>
  imageSize: ImageSize
  stride: number
  slotCount: number
}

type TrackedBox = { slotIndex: number; robot: Robot; box: LabelBox }

async function importReview(context: LegacyContext): Promise<Sidecar['review']> {
  const legacy = await readJsonFile(
    join(context.paths.datasetRoot, 'review.json'),
    legacyReviewSchema,
  )
  const review: Sidecar['review'] = {}
  for (const [stem, status] of Object.entries(legacy ?? {})) {
    const frame = parseFrameFile(`${stem}.jpg`)
    if (frame?.clip !== context.location.clip || status === 'unreviewed') continue
    review[String(frame.index)] = status
  }
  return review
}

/**
 * `mark_rounds.py` wrote two formats. The current one records `stride` and counts in
 * annotation frames, which are slots. The first one has no `stride` and counts native video
 * frames, so those are divided down. Rounds that still fall outside the clip are dropped:
 * one out-of-range round would make every later timeline save fail validation.
 */
async function importRounds(
  config: StudioConfig,
  context: LegacyContext,
): Promise<Sidecar['rounds']> {
  const marks = await readJsonFile(
    join(config.paths.annotations, 'round_marks', `${context.location.clip}.json`),
    legacyRoundMarksSchema,
  )
  const framesPerSlot = marks?.stride === undefined ? context.stride : 1
  return (marks?.rounds ?? [])
    .map(([start, end]) => ({
      start: Math.round(start / framesPerSlot),
      end: Math.round(end / framesPerSlot),
    }))
    .filter((round) => round.end >= round.start && round.end < context.slotCount)
}

function parseTrackedBox(line: string, context: LegacyContext): TrackedBox | null {
  const fields = line.trim().split(',').map(Number)
  if (fields.length < IDENTITY_FIELDS || fields.some(Number.isNaN)) return null
  const [nativeFrame = 0, trackId = 0, left = 0, top = 0, width = 0, height = 0] = fields
  const robot = ROBOT_BY_TRACK_ID[trackId]
  if (robot === undefined) return null
  return {
    slotIndex: Math.round(nativeFrame / context.stride),
    robot,
    box: {
      cx: (left + width / 2) / context.imageSize.width,
      cy: (top + height / 2) / context.imageSize.height,
      w: width / context.imageSize.width,
      h: height / context.imageSize.height,
    },
  }
}

function identityOfFrame(boxes: LabelBox[], tracked: TrackedBox[]): (Robot | null)[] {
  return boxes.map((box) => {
    const [best] = tracked
      .map((candidate) => ({
        robot: candidate.robot,
        overlap: intersectionOverUnion(box, candidate.box),
      }))
      .filter((candidate) => candidate.overlap >= IDENTITY_MATCH_IOU)
      .toSorted((first, second) => second.overlap - first.overlap)
    return best?.robot ?? null
  })
}

async function importIdentity(context: LegacyContext): Promise<Sidecar['identity']> {
  if (IDENTITY_FILE_CLIP[context.location.dataset] !== context.location.clip) return {}
  const text = await readTextFile(join(context.paths.datasetRoot, 'identity.txt'))
  const tracked = (text ?? '')
    .split('\n')
    .map((line) => parseTrackedBox(line, context))
    .filter((box) => box !== null)
  const identity: Sidecar['identity'] = {}
  for (const [slotIndex, boxes] of context.labels) {
    const robots = identityOfFrame(
      boxes,
      tracked.filter((box) => box.slotIndex === slotIndex),
    )
    if (robots.some((robot) => robot !== null)) identity[String(slotIndex)] = robots
  }
  return identity
}

/**
 * Builds a sidecar from the annotation files that predate the studio: the web editor's
 * `review.json`, `mark_rounds.py` round marks and the MOT-style `identity.txt`. Used only
 * until the studio writes its own sidecar for the clip.
 */
export async function importLegacySidecar(
  config: StudioConfig,
  context: LegacyContext,
): Promise<Sidecar> {
  const [review, rounds, identity] = await Promise.all([
    importReview(context),
    importRounds(config, context),
    importIdentity(context),
  ])
  return { ...emptySidecar(), review, rounds, identity }
}
