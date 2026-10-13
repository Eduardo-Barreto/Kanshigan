import { z } from 'zod'

import { footageSourceSchema, safeNameSchema } from './clip'

const MAX_WINDOWS_PER_REQUEST = 60
const MAX_URL_LENGTH = 300
// prep_clips.py decodes a whole window into memory: a minute of 1080p60 is already ~22 GB.
const MAX_WINDOW_SECONDS = 60

/** Footage folders the pipeline knows how to cut from: `data/raw/br` and `data/raw/jp`. */
export const rawSourceSchema = footageSourceSchema.exclude(['unknown'])
export type RawSource = z.infer<typeof rawSourceSchema>

export const rawVideoSchema = z.object({
  source: rawSourceSchema,
  name: safeNameSchema,
  durationSeconds: z.number(),
  width: z.number().int(),
  height: z.number().int(),
  fps: z.number(),
  sizeBytes: z.number().int(),
})
export type RawVideo = z.infer<typeof rawVideoSchema>

export const rawVideoListSchema = z.object({ videos: z.array(rawVideoSchema) })

/** A stretch of a raw video, in seconds, that becomes one clip. */
const roundWindowSchema = z
  .object({ start: z.number().min(0), end: z.number().min(0) })
  .refine((window) => window.end > window.start, { message: 'window ends before it starts' })
  .refine((window) => window.end - window.start <= MAX_WINDOW_SECONDS, {
    message: `window is longer than ${MAX_WINDOW_SECONDS} s`,
  })
export type RoundWindow = z.infer<typeof roundWindowSchema>

const JOB_STATUSES = ['queued', 'running', 'succeeded', 'failed', 'cancelled'] as const
const jobStatusSchema = z.enum(JOB_STATUSES)
export type JobStatus = z.infer<typeof jobStatusSchema>

const JOB_KINDS = ['download', 'segment', 'annotate'] as const
const jobKindSchema = z.enum(JOB_KINDS)
export type JobKind = z.infer<typeof jobKindSchema>

export const jobSchema = z.object({
  id: z.string(),
  kind: jobKindSchema,
  title: z.string(),
  status: jobStatusSchema,
  video: z.object({ source: rawSourceSchema, name: safeNameSchema }).nullable(),
  createdAt: z.string(),
  finishedAt: z.string().nullable(),
  stepIndex: z.number().int(),
  stepCount: z.number().int(),
  stepLabel: z.string(),
  lastLine: z.string(),
  error: z.string().nullable(),
  roundWindows: z.array(roundWindowSchema).nullable(),
})
export type Job = z.infer<typeof jobSchema>

export const jobListSchema = z.object({ jobs: z.array(jobSchema) })

export const downloadRequestSchema = z.object({
  url: z.url().max(MAX_URL_LENGTH),
  source: rawSourceSchema,
})
export type DownloadRequest = z.infer<typeof downloadRequestSchema>

export const annotateRequestSchema = z.object({
  source: rawSourceSchema,
  video: safeNameSchema,
  dataset: safeNameSchema,
  clipPrefix: safeNameSchema,
  windows: z.array(roundWindowSchema).min(1).max(MAX_WINDOWS_PER_REQUEST),
  scoreThreshold: z.number().gt(0).lt(1).nullable(),
})
export type AnnotateRequest = z.infer<typeof annotateRequestSchema>
