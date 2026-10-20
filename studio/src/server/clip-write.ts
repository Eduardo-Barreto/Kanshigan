import { copyFile, mkdir, rename, rm } from 'node:fs/promises'
import { join } from 'node:path'

import type { Robot, SaveFrame, SaveTimeline } from '@/shared/clip'

import { frameStem, type FramePlacement } from './clip-layout'
import { slotCount, type LoadedClip } from './clip-read'
import { writeExports } from './exports'
import { writeFileAtomic } from './files'
import { serializeLabels } from './labels'
import { notFound, unprocessable } from './problem'
import { writeSidecar, type Sidecar } from './sidecar'
import { ensureSlotCache, slotFileName } from './video'

/** One frame to persist: its slot index plus the boxes and review status that replace what is on disk. */
export type FrameSave = SaveFrame & { index: number }

function imagePath(clip: LoadedClip, index: number, placement: FramePlacement): string {
  return join(clip.paths.imagesOf(placement), `${frameStem(clip.location.clip, index)}.jpg`)
}

function labelPath(clip: LoadedClip, index: number, placement: FramePlacement): string {
  return join(clip.paths.labelsOf(placement), `${frameStem(clip.location.clip, index)}.txt`)
}

async function cachedSlotPath(clip: LoadedClip, index: number): Promise<string> {
  if (!clip.canExtractFrames || clip.video === null || index >= slotCount(clip)) {
    throw notFound(`quadro ${index} não existe em ${clip.location.clip}`)
  }
  await ensureSlotCache(clip.video.path, clip.stride, clip.paths.frameCache)
  const path = join(clip.paths.frameCache, slotFileName(index))
  if (!(await Bun.file(path).exists())) {
    throw notFound(`quadro ${index} está além do fim do vídeo de ${clip.location.clip}`)
  }
  return path
}

/** Path of the image to show for a slot: the dataset's own file, else a frame decoded from the clip video. */
export async function resolveFrameImage(clip: LoadedClip, index: number): Promise<string> {
  const placement = clip.placements.get(index)
  return placement ? imagePath(clip, index, placement) : cachedSlotPath(clip, index)
}

async function placeImage(clip: LoadedClip, index: number, target: FramePlacement): Promise<void> {
  const current = clip.placements.get(index)
  if (current === target) return
  await mkdir(clip.paths.imagesOf(target), { recursive: true })
  if (current === undefined) {
    await copyFile(await cachedSlotPath(clip, index), imagePath(clip, index, target))
  } else {
    await rename(imagePath(clip, index, current), imagePath(clip, index, target))
  }
  clip.placements.set(index, target)
}

function recordInSidecar(sidecar: Sidecar, save: FrameSave): void {
  const key = String(save.index)
  sidecar.review[key] = save.review
  const robots: (Robot | null)[] = save.boxes.map((box) => box.robot)
  if (robots.some((robot) => robot !== null)) sidecar.identity[key] = robots
  else delete sidecar.identity[key]
}

function forgetInSidecar(sidecar: Sidecar, index: number): void {
  delete sidecar.review[String(index)]
  delete sidecar.identity[String(index)]
}

async function saveFrame(clip: LoadedClip, save: FrameSave): Promise<void> {
  const isUntouchedSlot =
    !clip.placements.has(save.index) && save.boxes.length === 0 && save.review === 'pending'
  if (isUntouchedSlot) {
    forgetInSidecar(clip.sidecar, save.index)
    return
  }
  const target: FramePlacement = save.review === 'rejected' ? 'rejected' : 'active'
  const previous = clip.placements.get(save.index)
  await placeImage(clip, save.index, target)
  await writeFileAtomic(labelPath(clip, save.index, target), serializeLabels(save.boxes))
  // The old label goes only after the new one exists, so a crash never leaves an image unlabeled.
  if (previous !== undefined && previous !== target) {
    await rm(labelPath(clip, save.index, previous), { force: true })
  }
  clip.labels.set(save.index, save.boxes)
  recordInSidecar(clip.sidecar, save)
}

/**
 * Persists frames in order: writes each label file, moves the image between the dataset
 * and its `rejected` folder to match the review status, and updates the sidecar once.
 * A rejected frame leaves the dataset's `images/`, so training never reads it.
 */
export async function saveFrames(clip: LoadedClip, saves: FrameSave[]): Promise<void> {
  try {
    for (const save of saves) {
      // oxlint-disable-next-line no-await-in-loop -- sequential on purpose: each save mutates the shared clip
      await saveFrame(clip, save)
    }
  } finally {
    // Also on failure: frames saved before the one that threw already have new label
    // files, and their review and identity must not be left behind in the old sidecar.
    await writeSidecar(clip.paths.sidecar, clip.sidecar)
    await writeExports(clip)
  }
}

/** Replaces a clip's events and rounds, rejecting any that point past its last slot. */
export async function saveTimeline(clip: LoadedClip, timeline: SaveTimeline): Promise<void> {
  const lastSlot = slotCount(clip) - 1
  const indices = [
    ...timeline.events.map((event) => event.frameIndex),
    ...timeline.rounds.flatMap((round) => [round.start, round.end]),
  ]
  if (indices.some((index) => index > lastSlot)) {
    throw unprocessable(`o clip tem ${lastSlot + 1} quadros; há marcações depois do fim`)
  }
  clip.sidecar.events = timeline.events.toSorted(
    (first, second) => first.frameIndex - second.frameIndex,
  )
  clip.sidecar.rounds = timeline.rounds.toSorted((first, second) => first.start - second.start)
  await writeSidecar(clip.paths.sidecar, clip.sidecar)
  await writeExports(clip)
}
