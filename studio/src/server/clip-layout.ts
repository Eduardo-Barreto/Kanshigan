import { join } from 'node:path'

import type { FootageSource } from '@/shared/clip'

import type { StudioConfig } from './config'
import { listDirectory } from './files'

const FRAME_STEM = /^(?<clip>.+)_(?<index>\d+)$/
const IMAGE_EXTENSION = '.jpg'
const STEM_INDEX_DIGITS = 4
const VIDEO_SUBSETS: FootageSource[] = ['br', 'jp']

/** Identifies one clip inside one dataset directory. */
export type ClipLocation = { dataset: string; clip: string }

/** Where a frame's image and label live: in the dataset proper or set aside as rejected. */
export type FramePlacement = 'active' | 'rejected'

/** Absolute paths of everything the studio reads or writes for one clip. */
export type ClipPaths = {
  datasetRoot: string
  sidecar: string
  frameCache: string
  imagesOf: (placement: FramePlacement) => string
  labelsOf: (placement: FramePlacement) => string
}

/** File stem of a frame: the clip name plus its zero-padded slot index. */
export function frameStem(clip: string, index: number): string {
  return `${clip}_${String(index).padStart(STEM_INDEX_DIGITS, '0')}`
}

/** Splits an image file name into clip and slot index, or null when it is not a frame. */
export function parseFrameFile(fileName: string): { clip: string; index: number } | null {
  if (!fileName.endsWith(IMAGE_EXTENSION)) return null
  const groups = FRAME_STEM.exec(fileName.slice(0, -IMAGE_EXTENSION.length))?.groups
  if (groups?.clip === undefined || groups.index === undefined) return null
  return { clip: groups.clip, index: Number(groups.index) }
}

/** Resolves the directories a clip's frames, labels, sidecar and cache live in. */
export function clipPaths(config: StudioConfig, location: ClipLocation): ClipPaths {
  const datasetRoot = join(config.paths.annotations, location.dataset)
  const placementRoot = (placement: FramePlacement) =>
    placement === 'active' ? datasetRoot : join(datasetRoot, 'rejected')
  return {
    datasetRoot,
    sidecar: join(datasetRoot, 'studio', `${location.clip}.json`),
    frameCache: join(config.paths.frameCache, location.clip),
    imagesOf: (placement) => join(placementRoot(placement), 'images'),
    labelsOf: (placement) => join(placementRoot(placement), 'labels'),
  }
}

/** Finds a clip's native video and, from the folder it sits in, which footage it came from. */
export async function findClipVideo(
  config: StudioConfig,
  clip: string,
): Promise<{ path: string; source: FootageSource } | null> {
  for (const source of VIDEO_SUBSETS) {
    const path = join(config.paths.clipVideos, source, `${clip}.mp4`)
    // oxlint-disable-next-line no-await-in-loop -- two candidates, checked in priority order
    if (await Bun.file(path).exists()) return { path, source }
  }
  return null
}

/** Slot indices that have an image on disk, grouped by clip, for one placement of a dataset. */
export async function listFrameIndices(imagesDirectory: string): Promise<Map<string, number[]>> {
  const indicesByClip = new Map<string, number[]>()
  for (const fileName of await listDirectory(imagesDirectory)) {
    const frame = parseFrameFile(fileName)
    if (frame === null) continue
    const indices = indicesByClip.get(frame.clip) ?? []
    indices.push(frame.index)
    indicesByClip.set(frame.clip, indices)
  }
  return indicesByClip
}
