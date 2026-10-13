import { join } from 'node:path'

import type { ClipDetail, EventKind, Robot } from '@/shared/clip'

import { toDetail, type LoadedClip } from './clip-read'
import { writeFileAtomic } from './files'

const MS_PER_SECOND = 1000
const ANNOTATION_FPS = 10
const TRACK_ID_BY_ROBOT: Record<Robot, number> = { A: 0, B: 1 }
const PIXEL_DECIMALS = 1
const JSON_INDENT = 2

/** MOT-style identity lines, `nativeFrame,trackId,left,top,width,height`, as `evaluate.py --gold-mot` reads them. */
export function identityLines(detail: ClipDetail): string {
  return detail.frames
    .filter((frame) => frame.review !== 'rejected')
    .flatMap((frame) =>
      frame.boxes
        .filter((box) => box.robot !== null)
        .map((box) => {
          const width = box.w * detail.width
          const height = box.h * detail.height
          const left = box.cx * detail.width - width / 2
          const top = box.cy * detail.height - height / 2
          const trackId = box.robot === null ? 0 : TRACK_ID_BY_ROBOT[box.robot]
          const pixels = [left, top, width, height].map((value) => value.toFixed(PIXEL_DECIMALS))
          return [frame.index * detail.stride, trackId, ...pixels].join(',')
        }),
    )
    .map((line) => `${line}\n`)
    .join('')
}

/** Event times in milliseconds grouped by kind, the shape `evaluate.py --gold-events` reads. */
export function eventTimes(detail: ClipDetail): Partial<Record<EventKind, number[]>> {
  const times: Partial<Record<EventKind, number[]>> = {}
  for (const event of detail.events) {
    const seconds =
      detail.fps > 0
        ? (event.frameIndex * detail.stride) / detail.fps
        : event.frameIndex / ANNOTATION_FPS
    const milliseconds = seconds * MS_PER_SECOND
    times[event.kind] = [...(times[event.kind] ?? []), Math.round(milliseconds)]
  }
  return times
}

/**
 * Rewrites the files the evaluation scripts read, derived from the clip's current state:
 * identity tracks, event times and round limits. They live under `<dataset>/exports/` and
 * are regenerated on every save, so they never drift from what the studio shows.
 */
export async function writeExports(clip: LoadedClip): Promise<void> {
  const detail = toDetail(clip)
  const directory = join(clip.paths.datasetRoot, 'exports')
  const rounds = {
    clip: detail.clip,
    fps: detail.fps,
    stride: detail.stride,
    rounds: detail.rounds.map((round) => [round.start, round.end]),
  }
  await Promise.all([
    writeFileAtomic(join(directory, `${detail.clip}.identity.txt`), identityLines(detail)),
    writeFileAtomic(
      join(directory, `${detail.clip}.events.json`),
      `${JSON.stringify(eventTimes(detail), null, JSON_INDENT)}\n`,
    ),
    writeFileAtomic(
      join(directory, `${detail.clip}.rounds.json`),
      `${JSON.stringify(rounds, null, JSON_INDENT)}\n`,
    ),
  ])
}
