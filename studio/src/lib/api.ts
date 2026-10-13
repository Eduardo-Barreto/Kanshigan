import { queryOptions } from '@tanstack/react-query'
import { hc } from 'hono/client'
import { z } from 'zod'

import type { StudioApi } from '@/server/app'
import { sessionSchema, type Session } from '@/shared/auth'
import {
  clipDetailSchema,
  clipLocationSchema,
  librarySchema,
  moveOptionsSchema,
  type ClipDetail,
  type Library,
  type MoveOptions,
  type SaveFrame,
  type SaveTimeline,
} from '@/shared/clip'
import {
  jobListSchema,
  jobSchema,
  rawVideoListSchema,
  rawVideoSchema,
  type AnnotateRequest,
  type DownloadRequest,
  type Job,
  type RawSource,
  type RawVideo,
} from '@/shared/ingest'

const client = hc<StudioApi>('/')
const problemSchema = z.object({ error: z.string() })

/** Identifies a clip in the API: the dataset directory plus the clip name. */
export type ClipLocation = { dataset: string; clip: string }

/** A failed API call, carrying the server's own message so the UI can show it as is. */
class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function failure(response: Response): Promise<ApiError> {
  const body: unknown = await response.json().catch(() => null)
  const problem = problemSchema.safeParse(body)
  return new ApiError(
    response.status,
    problem.success ? problem.data.error : `o servidor respondeu ${response.status}`,
  )
}

async function readAs<Schema extends z.ZodType>(
  response: Response,
  schema: Schema,
): Promise<z.infer<Schema>> {
  if (!response.ok) throw await failure(response)
  const body: unknown = await response.json()
  return schema.parse(body)
}

/** Cache keys for every server resource, so invalidation never depends on a retyped string. */
export const queryKeys = {
  library: ['library'] as const,
  clip: (location: ClipLocation) => ['clip', location.dataset, location.clip] as const,
  session: ['session'] as const,
  moveOptions: (location: ClipLocation) =>
    ['clip', location.dataset, location.clip, 'move-options'] as const,
  videos: ['videos'] as const,
  jobs: ['jobs'] as const,
}

/** The whole library: every dataset with its clip summaries. */
export const libraryQuery = queryOptions({
  queryKey: queryKeys.library,
  queryFn: async (): Promise<Library> => readAs(await client.api.library.$get(), librarySchema),
})

/** One clip with every frame slot, its events and its rounds. */
export function clipQuery(location: ClipLocation) {
  return queryOptions({
    queryKey: queryKeys.clip(location),
    queryFn: async (): Promise<ClipDetail> =>
      readAs(
        await client.api.clips[':dataset'][':clip'].$get({ param: location }),
        clipDetailSchema,
      ),
  })
}

/** URL of the image for one frame slot, served from the dataset or decoded from the clip video. */
export function frameImageUrl(location: ClipLocation, index: number): string {
  return `/api/clips/${location.dataset}/${location.clip}/frames/${index}/image`
}

// A keepalive request outlives the page, so a save fired while the tab closes still lands.
// Browsers cap keepalive bodies at 64 KB, which a bulk save can exceed.
const KEEPALIVE_MAX_FRAMES = 20

/** Persists frames and returns the clip as the server now sees it. */
export async function saveFrames(
  location: ClipLocation,
  frames: (SaveFrame & { index: number })[],
): Promise<ClipDetail> {
  const response = await client.api.clips[':dataset'][':clip'].frames.$put(
    { param: location, json: { frames } },
    { init: { keepalive: frames.length <= KEEPALIVE_MAX_FRAMES } },
  )
  return readAs(response, clipDetailSchema)
}

/** Replaces the clip's events and rounds. */
export async function saveTimeline(
  location: ClipLocation,
  timeline: SaveTimeline,
): Promise<ClipDetail> {
  const response = await client.api.clips[':dataset'][':clip'].timeline.$put({
    param: location,
    json: timeline,
  })
  return readAs(response, clipDetailSchema)
}

const JOB_POLL_INTERVAL_MS = 1500
const isActive = (job: Job) => job.status === 'queued' || job.status === 'running'

/** Raw videos available to cut clips from. */
export const videosQuery = queryOptions({
  queryKey: queryKeys.videos,
  queryFn: async (): Promise<RawVideo[]> => {
    const { videos } = await readAs(await client.api.videos.$get(), rawVideoListSchema)
    return videos
  },
})

/** Background jobs, polled while any of them is still queued or running. */
export const jobsQuery = queryOptions({
  queryKey: queryKeys.jobs,
  queryFn: async ({ client: queryClient }): Promise<Job[]> => {
    const { jobs } = await readAs(await client.api.jobs.$get(), jobListSchema)
    const wasActive = new Set(
      (queryClient.getQueryData<Job[]>(queryKeys.jobs) ?? []).filter(isActive).map((job) => job.id),
    )
    if (jobs.some((job) => wasActive.has(job.id) && !isActive(job))) {
      void queryClient.invalidateQueries({ queryKey: queryKeys.videos })
      void queryClient.invalidateQueries({ queryKey: queryKeys.library })
    }
    return jobs
  },
  refetchInterval: (query) => (query.state.data?.some(isActive) ? JOB_POLL_INTERVAL_MS : false),
})

/** URL a `<video>` element can play and seek a raw video from. */
export function rawVideoUrl(video: { source: RawSource; name: string }): string {
  return `/api/videos/${video.source}/${video.name}/stream`
}

/** Uploads a video file into a footage folder under the given name. */
export async function uploadVideo(
  target: { source: RawSource; name: string },
  file: File,
): Promise<RawVideo> {
  const response = await fetch(`/api/uploads/${target.source}/${target.name}`, {
    method: 'PUT',
    headers: { 'Content-Type': file.type || 'application/octet-stream' },
    body: file,
  })
  return readAs(response, rawVideoSchema)
}

/** Queues a YouTube download. */
export async function startDownload(request: DownloadRequest): Promise<Job> {
  return readAs(await client.api.videos.download.$post({ json: request }), jobSchema)
}

/** Queues round detection for a raw video. */
export async function startSegmentation(video: { source: RawSource; name: string }): Promise<Job> {
  const response = await client.api.videos[':source'][':name'].segment.$post({ param: video })
  return readAs(response, jobSchema)
}

/** Queues cutting, cropping and SAM 3 annotation for a set of round windows. */
export async function startAnnotation(request: AnnotateRequest): Promise<Job> {
  return readAs(await client.api.annotations.$post({ json: request }), jobSchema)
}

/** Cancels a queued or running job. */
export async function cancelJob(jobId: string): Promise<Job[]> {
  const response = await client.api.jobs[':id'].cancel.$post({ param: { id: jobId } })
  const { jobs } = await readAs(response, jobListSchema)
  return jobs
}

/** Whether this browser may use the API. Checked before any workspace route loads. */
export const sessionQuery = queryOptions({
  queryKey: queryKeys.session,
  queryFn: async (): Promise<Session> =>
    readAs(await client.api.auth.session.$get(), sessionSchema),
  staleTime: 0,
})

/** Exchanges the studio password for a cookie session. */
export async function login(password: string): Promise<Session> {
  return readAs(await client.api.auth.login.$post({ json: { password } }), sessionSchema)
}

/** Datasets a clip could move to, each with the reasons it cannot yet, if any. */
export function moveOptionsQuery(location: ClipLocation) {
  return queryOptions({
    queryKey: queryKeys.moveOptions(location),
    queryFn: async (): Promise<MoveOptions['options']> => {
      const response = await client.api.clips[':dataset'][':clip']['move-options'].$get({
        param: location,
      })
      return (await readAs(response, moveOptionsSchema)).options
    },
    staleTime: 0,
  })
}

/** Moves a clip to another dataset and returns where it now lives. */
export async function moveClip(
  location: ClipLocation,
  targetDataset: string,
): Promise<ClipLocation> {
  const response = await client.api.clips[':dataset'][':clip'].move.$post({
    param: location,
    json: { targetDataset },
  })
  return readAs(response, clipLocationSchema)
}
