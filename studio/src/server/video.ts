import { mkdir, stat } from 'node:fs/promises'
import { join } from 'node:path'

import { z } from 'zod'

import { withTiming } from './log'

const SLOT_FILE_DIGITS = 5
const JPEG_QUALITY = '2'
const COMPLETE_MARKER = '.complete'

const probeSchema = z.object({
  streams: z
    .array(
      z.object({
        width: z.number().int(),
        height: z.number().int(),
        r_frame_rate: z.string(),
        nb_frames: z.coerce.number().int().optional(),
        nb_read_packets: z.coerce.number().int().optional(),
      }),
    )
    .min(1),
})

/** What the studio needs to know about a clip's video to lay out its frame slots. */
export type VideoProbe = { width: number; height: number; fps: number; frameCount: number }

function parseFrameRate(rate: string): number {
  const [numerator, denominator] = rate.split('/').map(Number)
  if (numerator === undefined || !denominator) return Number(rate)
  return numerator / denominator
}

async function runTool(command: string[]): Promise<string> {
  const child = Bun.spawn(command, { stdout: 'pipe', stderr: 'pipe' })
  const [output, errors, exitCode] = await Promise.all([
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
    child.exited,
  ])
  if (exitCode !== 0) throw new Error(`${command[0]} exited ${exitCode}: ${errors.trim()}`)
  return output
}

const probeCache = new Map<string, Promise<VideoProbe>>()

async function probeUncached(path: string): Promise<VideoProbe> {
  const output = await runTool([
    'ffprobe',
    '-v',
    'error',
    '-select_streams',
    'v:0',
    '-count_packets',
    '-show_entries',
    'stream=width,height,r_frame_rate,nb_frames,nb_read_packets',
    '-of',
    'json',
    path,
  ])
  const parsed: unknown = JSON.parse(output)
  const stream = probeSchema.parse(parsed).streams[0]
  if (stream === undefined) throw new Error(`no video stream in ${path}`)
  return {
    width: stream.width,
    height: stream.height,
    fps: parseFrameRate(stream.r_frame_rate),
    frameCount: stream.nb_frames ?? stream.nb_read_packets ?? 0,
  }
}

/** Probes a video's size, frame rate and length, memoized per file version. */
export async function probeVideo(path: string): Promise<VideoProbe> {
  const { mtimeMs } = await stat(path)
  const cacheKey = `${path}:${mtimeMs}`
  const cached = probeCache.get(cacheKey)
  if (cached) return cached
  const probing = probeUncached(path)
  probeCache.set(cacheKey, probing)
  return probing
}

/** File name of a cached slot image. */
export function slotFileName(slotIndex: number): string {
  return `${String(slotIndex).padStart(SLOT_FILE_DIGITS, '0')}.jpg`
}

const extractions = new Map<string, Promise<void>>()

async function extractUncached(videoPath: string, stride: number, cacheDirectory: string) {
  await mkdir(cacheDirectory, { recursive: true })
  await withTiming('frames.extract', { videoPath, stride }, () =>
    runTool([
      'ffmpeg',
      '-y',
      '-loglevel',
      'error',
      '-i',
      videoPath,
      '-vf',
      `select=not(mod(n\\,${stride}))`,
      '-fps_mode',
      'vfr',
      '-q:v',
      JPEG_QUALITY,
      '-start_number',
      '0',
      join(cacheDirectory, `%0${SLOT_FILE_DIGITS}d.jpg`),
    ]),
  )
  await Bun.write(join(cacheDirectory, COMPLETE_MARKER), '')
}

/**
 * Decodes every `stride`-th frame of a clip into the cache, once. Slot `k` is native frame
 * `k * stride`, the same frame the SAM annotator dumps for label index `k`, so a cached
 * slot and a dataset image of the same index show the same instant.
 */
export async function ensureSlotCache(
  videoPath: string,
  stride: number,
  cacheDirectory: string,
): Promise<void> {
  const inFlight = extractions.get(cacheDirectory)
  if (inFlight) return inFlight
  if (await Bun.file(join(cacheDirectory, COMPLETE_MARKER)).exists()) return
  const extraction = extractUncached(videoPath, stride, cacheDirectory)
  extractions.set(cacheDirectory, extraction)
  try {
    await extraction
  } finally {
    extractions.delete(cacheDirectory)
  }
}
