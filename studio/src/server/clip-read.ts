import { basename, join } from 'node:path'

import type { Box, ClipDetail, ClipSummary, Frame, ReviewStatus, Role } from '@/shared/clip'

import {
  clipPaths,
  findClipVideo,
  frameStem,
  listFrameIndices,
  type ClipLocation,
  type ClipPaths,
  type FramePlacement,
} from './clip-layout'
import type { StudioConfig } from './config'
import { datasetProfile, defaultReview } from './datasets'
import { listDirectory, readTextFile } from './files'
import { readJpegSize, type ImageSize } from './jpeg-size'
import { parseLabels, type LabelBox } from './labels'
import { importLegacySidecar } from './legacy-import'
import { notFound } from './problem'
import { readSidecar, type Sidecar } from './sidecar'
import { probeVideo, type VideoProbe } from './video'

const ANNOTATION_FPS = 10
const DEFAULT_STRIDE = 6
const SIZE_TOLERANCE_PX = 2
const ROBOTS_PER_ROUND = 2

/** A clip as it sits on disk, read once and shared by the summary, detail and write paths. */
export type LoadedClip = {
  location: ClipLocation
  paths: ClipPaths
  role: Role
  video: (VideoProbe & { path: string }) | null
  source: ClipSummary['source']
  imageSize: ImageSize
  stride: number
  placements: Map<number, FramePlacement>
  labels: Map<number, LabelBox[]>
  sidecar: Sidecar
  hasOwnSidecar: boolean
  canExtractFrames: boolean
}

async function readPlacements(
  paths: ClipPaths,
  clip: string,
): Promise<Map<number, FramePlacement>> {
  const [active, rejected] = await Promise.all([
    listFrameIndices(paths.imagesOf('active')),
    listFrameIndices(paths.imagesOf('rejected')),
  ])
  const placements = new Map<number, FramePlacement>()
  for (const index of rejected.get(clip) ?? []) placements.set(index, 'rejected')
  for (const index of active.get(clip) ?? []) placements.set(index, 'active')
  return placements
}

async function readLabels(
  paths: ClipPaths,
  clip: string,
  placements: Map<number, FramePlacement>,
): Promise<Map<number, LabelBox[]>> {
  const entries = await Promise.all(
    [...placements].map(async ([index, placement]) => {
      const text = await readTextFile(
        join(paths.labelsOf(placement), `${frameStem(clip, index)}.txt`),
      )
      return [index, parseLabels(text ?? '')] as const
    }),
  )
  return new Map(entries)
}

async function readImageSize(
  paths: ClipPaths,
  clip: string,
  placements: Map<number, FramePlacement>,
): Promise<ImageSize | null> {
  const first = [...placements].at(0)
  if (first === undefined) return null
  const [index, placement] = first
  return readJpegSize(join(paths.imagesOf(placement), `${frameStem(clip, index)}.jpg`))
}

function sameSpace(video: VideoProbe, image: ImageSize): boolean {
  return (
    Math.abs(video.width - image.width) <= SIZE_TOLERANCE_PX &&
    Math.abs(video.height - image.height) <= SIZE_TOLERANCE_PX
  )
}

async function probeClipVideo(found: Awaited<ReturnType<typeof findClipVideo>>) {
  return found ? { ...(await probeVideo(found.path)), path: found.path } : null
}

function strideOf(video: VideoProbe | null): number {
  return video ? Math.max(1, Math.round(video.fps / ANNOTATION_FPS)) : DEFAULT_STRIDE
}

/** Reads a clip's frames, labels, video and sidecar from disk. Throws 404 when it has neither frames nor a sidecar. */
export async function loadClip(config: StudioConfig, location: ClipLocation): Promise<LoadedClip> {
  const paths = clipPaths(config, location)
  const [placements, ownSidecar, foundVideo] = await Promise.all([
    readPlacements(paths, location.clip),
    readSidecar(paths.sidecar),
    findClipVideo(config, location.clip),
  ])
  if (placements.size === 0 && ownSidecar === null) {
    throw notFound(`clip ${location.dataset}/${location.clip} não existe`)
  }
  const [video, labels, measuredSize] = await Promise.all([
    probeClipVideo(foundVideo),
    readLabels(paths, location.clip, placements),
    readImageSize(paths, location.clip, placements),
  ])
  const imageSize = measuredSize ?? video ?? { width: 0, height: 0 }
  const stride = strideOf(video)
  const canExtractFrames = video !== null && sameSpace(video, imageSize)
  const highestIndex = Math.max(-1, ...placements.keys())
  const videoSlots = canExtractFrames ? Math.ceil(video.frameCount / stride) : 0
  const importLegacy = () =>
    importLegacySidecar(config, {
      location,
      paths,
      labels,
      imageSize,
      stride,
      slotCount: Math.max(highestIndex + 1, videoSlots),
    })
  return {
    location,
    paths,
    role: datasetProfile(location.dataset).role,
    video,
    source: foundVideo?.source ?? 'unknown',
    imageSize,
    stride,
    placements,
    labels,
    sidecar: ownSidecar ?? (await importLegacy()),
    hasOwnSidecar: ownSidecar !== null,
    canExtractFrames,
  }
}

/** Number of annotation slots a clip has: one per `stride` native frames when frames can be extracted. */
export function slotCount(clip: LoadedClip): number {
  const highestIndex = Math.max(-1, ...clip.placements.keys())
  if (!clip.canExtractFrames || clip.video === null) return highestIndex + 1
  return Math.max(highestIndex + 1, Math.ceil(clip.video.frameCount / clip.stride))
}

/** Review status of a slot: what the sidecar records, else the dataset's default for labeled frames. */
export function reviewOf(clip: LoadedClip, index: number): ReviewStatus {
  const recorded = clip.sidecar.review[String(index)]
  if (recorded) return recorded
  return clip.placements.has(index) ? defaultReview(clip.role) : 'pending'
}

function boxesOf(clip: LoadedClip, index: number): Box[] {
  const labels = clip.labels.get(index) ?? []
  const robots = clip.sidecar.identity[String(index)]
  const aligned = robots?.length === labels.length ? robots : undefined
  return labels.map((label, position) => ({ ...label, robot: aligned?.[position] ?? null }))
}

function frameOf(clip: LoadedClip, index: number): Frame {
  return {
    index,
    hasImage: clip.placements.has(index),
    boxes: boxesOf(clip, index),
    review: reviewOf(clip, index),
  }
}

function framesOf(clip: LoadedClip): Frame[] {
  const indices = clip.canExtractFrames
    ? Array.from({ length: slotCount(clip) }, (_unused, index) => index)
    : [...clip.placements.keys()].toSorted((first, second) => first - second)
  return indices.map((index) => frameOf(clip, index))
}

function isIdentified(frame: Frame): boolean {
  return frame.boxes.length > 0 && frame.boxes.every((box) => box.robot !== null)
}

/** Aggregates a clip into the numbers the library shows. */
export function toSummary(clip: LoadedClip): ClipSummary {
  const labeled = framesOf(clip).filter((frame) => frame.hasImage)
  const kept = labeled.filter((frame) => frame.review !== 'rejected')
  return {
    ...clip.location,
    role: clip.role,
    source: clip.source,
    width: clip.imageSize.width,
    height: clip.imageSize.height,
    fps: clip.video?.fps ?? 0,
    frameCount: labeled.length,
    slotCount: slotCount(clip),
    boxCount: kept.reduce((total, frame) => total + frame.boxes.length, 0),
    approvedCount: labeled.filter((frame) => frame.review === 'approved').length,
    rejectedCount: labeled.length - kept.length,
    bothRobotsCount: kept.filter((frame) => frame.boxes.length === ROBOTS_PER_ROUND).length,
    identifiedCount: kept.filter(isIdentified).length,
    eventCount: clip.sidecar.events.length,
    roundCount: clip.sidecar.rounds.length,
    hasVideo: clip.video !== null,
    canExtractFrames: clip.canExtractFrames,
    thumbnailIndex: labeled.at(Math.floor(labeled.length / 2))?.index ?? null,
  }
}

/** Expands a clip into every frame slot plus its events and rounds, for the workbench. */
export function toDetail(clip: LoadedClip): ClipDetail {
  return {
    ...toSummary(clip),
    stride: clip.stride,
    frames: framesOf(clip),
    events: clip.sidecar.events,
    rounds: clip.sidecar.rounds,
  }
}

/** Clip names that belong to a dataset: any clip with a frame on disk or a studio sidecar. */
export async function listClipNames(config: StudioConfig, dataset: string): Promise<string[]> {
  const paths = clipPaths(config, { dataset, clip: '' })
  const [active, rejected, sidecarFiles] = await Promise.all([
    listFrameIndices(paths.imagesOf('active')),
    listFrameIndices(paths.imagesOf('rejected')),
    listDirectory(join(paths.datasetRoot, 'studio')),
  ])
  const fromSidecars = sidecarFiles
    .filter((fileName) => fileName.endsWith('.json'))
    .map((fileName) => basename(fileName, '.json'))
  return [...new Set([...active.keys(), ...rejected.keys(), ...fromSidecars])].toSorted()
}
