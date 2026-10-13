import { mkdir, rename, rm, stat } from 'node:fs/promises'
import { join } from 'node:path'

import { safeNameSchema } from '@/shared/clip'
import type {
  AnnotateRequest,
  DownloadRequest,
  RawSource,
  RawVideo,
  RoundWindow,
} from '@/shared/ingest'

import { clipPaths, findClipVideo } from './clip-layout'
import type { StudioConfig } from './config'
import { datasetProfile } from './datasets'
import { listDirectory } from './files'
import type { JobSpec, JobStep } from './jobs'
import { conflict, notFound, unprocessable } from './problem'
import { runProcess } from './process'
import { emptySidecar, readSidecar, writeSidecar } from './sidecar'
import { probeVideo } from './video'

const RAW_SOURCES: RawSource[] = ['br', 'jp']
const VIDEO_EXTENSION = '.mp4'
const YOUTUBE_HOSTS = new Set(['youtube.com', 'www.youtube.com', 'm.youtube.com', 'youtu.be'])
const YOUTUBE_VIDEO_ID = /^[A-Za-z0-9_-]{11}$/
const MAX_DOWNLOAD_SIZE = '2G'
const YOUTUBE_FORMAT = 'bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b'
const ROUND_LINE = /^round \d+: (?<start>[\d.]+)-(?<end>[\d.]+)/
const CLIP_NUMBER_DIGITS = 2
const SAM_PROMPT = 'toy'
const FROZEN_ROLES = new Set(['gold', 'train', 'val'])

function pipelineEnvironment(config: StudioConfig): Record<string, string> {
  return { KANSHIGAN_DATA_ROOT: config.dataRoot }
}

function rawVideoPath(config: StudioConfig, source: RawSource, name: string): string {
  return join(config.paths.rawVideos, source, `${name}${VIDEO_EXTENSION}`)
}

async function describeRawVideo(
  config: StudioConfig,
  source: RawSource,
  name: string,
): Promise<RawVideo> {
  const path = rawVideoPath(config, source, name)
  const [probe, file] = await Promise.all([probeVideo(path), stat(path)])
  return {
    source,
    name,
    durationSeconds: probe.fps > 0 ? probe.frameCount / probe.fps : 0,
    width: probe.width,
    height: probe.height,
    fps: probe.fps,
    sizeBytes: file.size,
  }
}

/** Lists the raw videos available to cut clips from. A file ffprobe cannot read is left out. */
export async function listRawVideos(config: StudioConfig): Promise<RawVideo[]> {
  const perSource = await Promise.all(
    RAW_SOURCES.map(async (source) => {
      const names = (await listDirectory(join(config.paths.rawVideos, source)))
        .filter((fileName) => fileName.endsWith(VIDEO_EXTENSION))
        .map((fileName) => fileName.slice(0, -VIDEO_EXTENSION.length))
        .filter((name) => safeNameSchema.safeParse(name).success)
      const described = await Promise.all(
        names.toSorted().map((name) => describeRawVideo(config, source, name).catch(() => null)),
      )
      return described.filter((video) => video !== null)
    }),
  )
  return perSource.flat()
}

/** Resolves a raw video to its path, or throws 404 when it is not on disk. */
export async function requireRawVideo(
  config: StudioConfig,
  source: RawSource,
  name: string,
): Promise<string> {
  const path = rawVideoPath(config, source, name)
  if (!(await Bun.file(path).exists())) throw notFound(`vídeo ${source}/${name} não existe`)
  return path
}

/**
 * Stores an uploaded video under `data/raw/<source>`. The body is written to a temp file
 * and only renamed into place once ffprobe confirms it holds a video stream.
 */
export async function storeUpload(
  config: StudioConfig,
  target: { source: RawSource; name: string },
  body: ReadableStream<Uint8Array>,
): Promise<RawVideo> {
  const path = rawVideoPath(config, target.source, target.name)
  if (await Bun.file(path).exists()) {
    throw conflict(`já existe um vídeo ${target.source}/${target.name}`)
  }
  await mkdir(join(config.paths.rawVideos, target.source), { recursive: true })
  const incomingPath = `${path}.${crypto.randomUUID()}.incoming`
  try {
    await Bun.write(incomingPath, new Response(body))
    await probeVideo(incomingPath)
  } catch {
    await rm(incomingPath, { force: true })
    throw unprocessable('o arquivo enviado não é um vídeo que o ffmpeg consiga ler')
  }
  await rename(incomingPath, path)
  return describeRawVideo(config, target.source, target.name)
}

/**
 * Rebuilds a watch URL from the video id alone. The user's string never reaches yt-dlp:
 * its URL parser disagrees with WHATWG's on inputs like `http://youtube.com\@host/`, and a
 * channel or playlist link would download everything in it.
 */
export function canonicalYoutubeUrl(input: string): string {
  const url = new URL(input)
  if (url.protocol !== 'https:' || !YOUTUBE_HOSTS.has(url.hostname)) {
    throw unprocessable('só links https do YouTube são aceitos')
  }
  const videoId =
    url.hostname === 'youtu.be'
      ? url.pathname.slice(1)
      : (url.searchParams.get('v') ?? url.pathname.replace(/^\/(?:shorts|live|embed)\//, ''))
  if (!YOUTUBE_VIDEO_ID.test(videoId)) throw unprocessable('o link não aponta para um vídeo')
  return `https://www.youtube.com/watch?v=${videoId}`
}

/** Builds the job that downloads one YouTube video into the raw folder with yt-dlp. */
export function downloadJob(config: StudioConfig, request: DownloadRequest): JobSpec {
  const url = canonicalYoutubeUrl(request.url)
  const directory = join(config.paths.rawVideos, request.source)
  return {
    kind: 'download',
    title: `Baixar ${url}`,
    video: null,
    steps: [
      {
        label: 'Baixar do YouTube',
        run: async ({ emit, signal }) => {
          await mkdir(directory, { recursive: true })
          await runProcess(
            [
              'yt-dlp',
              '--no-playlist',
              '--use-extractors',
              'youtube',
              '--max-downloads',
              '1',
              '--match-filter',
              '!is_live',
              '--max-filesize',
              MAX_DOWNLOAD_SIZE,
              '--newline',
              '--format',
              YOUTUBE_FORMAT,
              '--merge-output-format',
              'mp4',
              '--output',
              join(directory, 'yt_%(id)s.%(ext)s'),
              '--',
              url,
            ],
            { cwd: directory, onLine: emit, signal },
          )
        },
      },
    ],
  }
}

function parseRoundLine(line: string): RoundWindow | null {
  const groups = ROUND_LINE.exec(line.trim())?.groups
  if (groups?.start === undefined || groups.end === undefined) return null
  return { start: Number(groups.start), end: Number(groups.end) }
}

/** Builds the job that proposes round windows for a raw video with `segment_rounds.py`. */
export function segmentJob(
  config: StudioConfig,
  video: { source: RawSource; name: string; path: string },
): JobSpec {
  return {
    kind: 'segment',
    title: `Detectar rounds de ${video.name}`,
    video: { source: video.source, name: video.name },
    steps: [
      {
        label: 'Procurar rounds por movimento e presença do dohyo',
        run: async ({ emit, signal, setRoundWindows }) => {
          const windows: RoundWindow[] = []
          await runProcess(['uv', 'run', 'python', 'segment_rounds.py', video.path], {
            cwd: join(config.repoRoot, 'experiments', 'pre-banca'),
            signal,
            onLine: (line) => {
              emit(line)
              const window = parseRoundLine(line)
              if (window !== null) windows.push(window)
            },
          })
          setRoundWindows(windows)
        },
      },
    ],
  }
}

function clipNames(request: AnnotateRequest): string[] {
  return request.windows.map(
    (_window, position) =>
      `${request.clipPrefix}_r${String(position + 1).padStart(CLIP_NUMBER_DIGITS, '0')}`,
  )
}

async function assertClipNamesAreFree(config: StudioConfig, names: string[]): Promise<void> {
  const existing = await Promise.all(names.map((name) => findClipVideo(config, name)))
  const taken = names.filter((_name, position) => existing[position] !== null)
  if (taken.length > 0) throw conflict(`já existem clips com estes nomes: ${taken.join(', ')}`)
}

function cutStep(config: StudioConfig, request: AnnotateRequest, videoPath: string): JobStep {
  const clips = clipNames(request).map((id, position) => ({
    id,
    source: videoPath,
    subset: request.source,
    start: request.windows[position]?.start,
    end: request.windows[position]?.end,
  }))
  return {
    label: 'Cortar os rounds e recortar no dohyo',
    run: async ({ emit, signal }) => {
      // Checked again here: another job queued earlier may have taken the names since.
      await assertClipNamesAreFree(config, clipNames(request))
      await mkdir(config.paths.jobLogs, { recursive: true })
      const specPath = join(config.paths.jobLogs, `${crypto.randomUUID()}-clips.yaml`)
      await Bun.write(specPath, JSON.stringify({ clips }))
      await runProcess(['uv', 'run', 'python', 'prep_clips.py', '--config', specPath], {
        cwd: join(config.repoRoot, 'experiments', 'pre-banca'),
        env: pipelineEnvironment(config),
        onLine: emit,
        signal,
      })
      // Registered right after cutting, so a clip whose SAM run fails or is cancelled
      // still shows up in the library instead of sitting on disk with its name taken.
      await registerClips(config, request)
    },
  }
}

function samStep(config: StudioConfig, request: AnnotateRequest, clip: string): JobStep {
  const threshold =
    request.scoreThreshold === null ? [] : ['--score-thresh', String(request.scoreThreshold)]
  return {
    label: `SAM 3 em ${clip}`,
    run: async ({ emit, signal }) => {
      await runProcess(
        [
          'uv',
          'run',
          'python',
          'annotate_to_yolo.py',
          clip,
          '--split',
          request.dataset,
          '--prompt',
          SAM_PROMPT,
          ...threshold,
        ],
        {
          cwd: join(config.repoRoot, 'experiments', 'sam3-poc'),
          env: pipelineEnvironment(config),
          onLine: emit,
          signal,
        },
      )
    },
  }
}

async function registerClips(config: StudioConfig, request: AnnotateRequest): Promise<void> {
  await Promise.all(
    clipNames(request).map(async (clip) => {
      const { sidecar } = clipPaths(config, { dataset: request.dataset, clip })
      if ((await readSidecar(sidecar)) === null) await writeSidecar(sidecar, emptySidecar())
    }),
  )
}

/**
 * Builds the job that turns round windows of a raw video into annotated clips: cut and crop
 * to the dohyo and register each clip, then propose boxes with SAM 3. Frozen datasets (gold, train, val) are refused as targets.
 */
export async function annotateJob(
  config: StudioConfig,
  request: AnnotateRequest,
): Promise<JobSpec> {
  if (FROZEN_ROLES.has(datasetProfile(request.dataset).role)) {
    throw unprocessable(
      `${request.dataset} é um conjunto congelado; anote num conjunto de candidatos`,
    )
  }
  const videoPath = await requireRawVideo(config, request.source, request.video)
  const names = clipNames(request)
  await assertClipNamesAreFree(config, names)
  return {
    kind: 'annotate',
    title: `Anotar ${names.length} rounds de ${request.video}`,
    video: { source: request.source, name: request.video },
    steps: [
      cutStep(config, request, videoPath),
      ...names.map((clip) => samStep(config, request, clip)),
    ],
  }
}
