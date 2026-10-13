import { zValidator } from '@hono/zod-validator'
import { Hono } from 'hono'
import { bodyLimit } from 'hono/body-limit'
import { csrf } from 'hono/csrf'
import { requestId } from 'hono/request-id'
import { secureHeaders } from 'hono/secure-headers'
import { z } from 'zod'

import { loginRequestSchema } from '@/shared/auth'
import { moveClipSchema, safeNameSchema, saveFrameSchema, saveTimelineSchema } from '@/shared/clip'
import { annotateRequestSchema, downloadRequestSchema, rawSourceSchema } from '@/shared/ingest'

import { createAuth, hostGuard } from './auth'
import { moveBlockers, moveClip } from './clip-move'
import { loadClip, toDetail } from './clip-read'
import { resolveFrameImage, saveFrames, saveTimeline } from './clip-write'
import type { StudioConfig } from './config'
import { datasetProfile } from './datasets'
import { writeExports } from './exports'
import {
  annotateJob,
  downloadJob,
  listRawVideos,
  requireRawVideo,
  segmentJob,
  storeUpload,
} from './ingest'
import { createJobQueue, type JobQueue } from './jobs'
import { readLibrary } from './library'
import { log } from './log'
import { Problem, notFound, unprocessable } from './problem'
import { serveWithRange } from './range'
import { serialize } from './serialize'

const JSON_BODY_LIMIT_BYTES = 1_048_576
const MAX_FRAMES_PER_SAVE = 2000
const HTTP_BAD_REQUEST = 400
const HTTP_SERVER_ERROR = 500
const FRAME_IMAGE_CACHE_CONTROL = 'private, max-age=3600'
/** Largest video upload accepted, enforced both by the route and by the HTTP server. */
export const MAX_UPLOAD_BYTES = 4_294_967_296
const HTTP_ACCEPTED = 202
const HTTP_CREATED = 201
const MOVE_TARGETS = ['gold_crop', 'gold_jp', 'gold_candidates', 'train', 'val']

const locationSchema = z.object({ dataset: safeNameSchema, clip: safeNameSchema })
const frameLocationSchema = locationSchema.extend({ index: z.coerce.number().int().min(0) })
const saveFramesSchema = z.object({
  frames: z
    .array(saveFrameSchema.extend({ index: z.number().int().min(0) }))
    .min(1)
    .max(MAX_FRAMES_PER_SAVE),
})

function clipKey(location: z.infer<typeof locationSchema>): string {
  return `${location.dataset}/${location.clip}`
}

function clipRoutes(config: StudioConfig) {
  return new Hono()
    .get('/library', async (context) => context.json(await readLibrary(config)))
    .get('/clips/:dataset/:clip', zValidator('param', locationSchema), async (context) => {
      const clip = await loadClip(config, context.req.valid('param'))
      return context.json(toDetail(clip))
    })
    .get(
      '/clips/:dataset/:clip/frames/:index/image',
      zValidator('param', frameLocationSchema),
      async (context) => {
        const { index, ...location } = context.req.valid('param')
        const clip = await loadClip(config, location)
        const image = Bun.file(await resolveFrameImage(clip, index))
        return new Response(image, {
          headers: { 'Content-Type': 'image/jpeg', 'Cache-Control': FRAME_IMAGE_CACHE_CONTROL },
        })
      },
    )
    .put(
      '/clips/:dataset/:clip/frames',
      zValidator('param', locationSchema),
      zValidator('json', saveFramesSchema),
      async (context) => {
        const location = context.req.valid('param')
        const { frames } = context.req.valid('json')
        const detail = await serialize(clipKey(location), async () => {
          const clip = await loadClip(config, location)
          await saveFrames(clip, frames)
          return toDetail(clip)
        })
        return context.json(detail)
      },
    )
    .put(
      '/clips/:dataset/:clip/timeline',
      zValidator('param', locationSchema),
      zValidator('json', saveTimelineSchema),
      async (context) => {
        const location = context.req.valid('param')
        const timeline = context.req.valid('json')
        const detail = await serialize(clipKey(location), async () => {
          const clip = await loadClip(config, location)
          await saveTimeline(clip, timeline)
          return toDetail(clip)
        })
        return context.json(detail)
      },
    )
    .get(
      '/clips/:dataset/:clip/move-options',
      zValidator('param', locationSchema),
      async (context) => {
        const location = context.req.valid('param')
        const [clip, library] = await Promise.all([loadClip(config, location), readLibrary(config)])
        const candidates = new Set([
          ...library.datasets.map((dataset) => dataset.dataset),
          ...MOVE_TARGETS,
        ])
        candidates.delete(location.dataset)
        const options = await Promise.all(
          [...candidates].map(async (dataset) => ({
            dataset,
            label: datasetProfile(dataset).label,
            role: datasetProfile(dataset).role,
            blockers: await moveBlockers(clip, dataset),
          })),
        )
        return context.json({ options })
      },
    )
    .post(
      '/clips/:dataset/:clip/move',
      zValidator('param', locationSchema),
      zValidator('json', moveClipSchema),
      async (context) => {
        const location = context.req.valid('param')
        const { targetDataset } = context.req.valid('json')
        const destination = await serialize(clipKey(location), async () =>
          moveClip(config, await loadClip(config, location), targetDataset),
        )
        await writeExports(await loadClip(config, destination))
        return context.json(destination)
      },
    )
}

const videoLocationSchema = z.object({ source: rawSourceSchema, name: safeNameSchema })

function ingestRoutes(config: StudioConfig, jobs: JobQueue) {
  return new Hono()
    .get('/videos', async (context) => context.json({ videos: await listRawVideos(config) }))
    .get(
      '/videos/:source/:name/stream',
      zValidator('param', videoLocationSchema),
      async (context) => {
        const { source, name } = context.req.valid('param')
        const path = await requireRawVideo(config, source, name)
        return serveWithRange(path, context.req.header('Range'), 'video/mp4')
      },
    )
    .post(
      '/videos/:source/:name/segment',
      zValidator('param', videoLocationSchema),
      async (context) => {
        const video = context.req.valid('param')
        const path = await requireRawVideo(config, video.source, video.name)
        return context.json(jobs.enqueue(segmentJob(config, { ...video, path })), HTTP_ACCEPTED)
      },
    )
    .post('/videos/download', zValidator('json', downloadRequestSchema), (context) =>
      context.json(jobs.enqueue(downloadJob(config, context.req.valid('json'))), HTTP_ACCEPTED),
    )
    .post('/annotations', zValidator('json', annotateRequestSchema), async (context) => {
      const spec = await annotateJob(config, context.req.valid('json'))
      return context.json(jobs.enqueue(spec), HTTP_ACCEPTED)
    })
    .get('/jobs', (context) => context.json({ jobs: jobs.list() }))
    .post('/jobs/:id/cancel', zValidator('param', z.object({ id: z.uuid() })), (context) => {
      if (!jobs.cancel(context.req.valid('param').id)) throw notFound('tarefa não encontrada')
      return context.json({ jobs: jobs.list() })
    })
}

function uploadRoutes(config: StudioConfig) {
  return new Hono().put(
    '/uploads/:source/:name',
    zValidator('param', videoLocationSchema),
    bodyLimit({ maxSize: MAX_UPLOAD_BYTES }),
    async (context) => {
      const body = context.req.raw.body
      if (body === null || !context.req.header('Content-Length')) {
        throw unprocessable('o envio precisa declarar o tamanho (Content-Length)')
      }
      const video = await storeUpload(config, context.req.valid('param'), body)
      return context.json(video, HTTP_CREATED)
    },
  )
}

function authRoutes(auth: ReturnType<typeof createAuth>) {
  return new Hono()
    .get('/auth/session', async (context) => context.json(await auth.sessionOf(context)))
    .post('/auth/login', zValidator('json', loginRequestSchema), async (context) =>
      context.json(await auth.login(context, context.req.valid('json').password)),
    )
    .post('/auth/logout', (context) => context.json(auth.logout(context)))
}

/** Builds the HTTP API. Routes are chained so the client derives its types from this value. */
export function createApp(
  config: StudioConfig,
  jobs: JobQueue = createJobQueue(config.paths.jobLogs, [config.dataRoot, config.repoRoot]),
) {
  const auth = createAuth(config.password)
  const app = new Hono()
    .use(requestId())
    .use(secureHeaders())
    .use(async (context, next) => {
      const startedAt = performance.now()
      await next()
      log.info('http.request', {
        requestId: context.get('requestId'),
        method: context.req.method,
        path: context.req.path,
        status: context.res.status,
        durationMs: Math.round(performance.now() - startedAt),
      })
    })
    .use('/api/*', hostGuard)
    .use('/api/*', csrf())
    .use('/api/auth/*', bodyLimit({ maxSize: JSON_BODY_LIMIT_BYTES }))
    .route('/api', authRoutes(auth))
    .use('/api/*', auth.guard)
    .use('/api/clips/*', bodyLimit({ maxSize: JSON_BODY_LIMIT_BYTES }))
    .use('/api/videos/*', bodyLimit({ maxSize: JSON_BODY_LIMIT_BYTES }))
    .use('/api/annotations', bodyLimit({ maxSize: JSON_BODY_LIMIT_BYTES }))
    .route('/api', clipRoutes(config))
    .route('/api', ingestRoutes(config, jobs))
    .route('/api', uploadRoutes(config))

  app.onError((error, context) => {
    if (error instanceof Problem) return context.json({ error: error.message }, error.status)
    if (error instanceof z.ZodError) {
      return context.json({ error: 'dados inválidos em disco ou na requisição' }, HTTP_BAD_REQUEST)
    }
    log.error('http.unhandled', { path: context.req.path, error: error.message })
    return context.json({ error: 'erro interno do servidor' }, HTTP_SERVER_ERROR)
  })
  return app
}

/** The API's route types, consumed by the typed client. */
export type StudioApi = ReturnType<typeof createApp>
