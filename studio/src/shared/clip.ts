import { z } from 'zod'

const SAFE_NAME = /^[A-Za-z0-9][A-Za-z0-9_-]*$/
const MAX_NAME_LENGTH = 80

/** A directory-safe identifier: dataset and clip names become path segments on the server. */
export const safeNameSchema = z.string().regex(SAFE_NAME).max(MAX_NAME_LENGTH)

const ROLES = ['gold', 'gold_candidate', 'train', 'val', 'other'] as const
const roleSchema = z.enum(ROLES)
export type Role = z.infer<typeof roleSchema>

const FOOTAGE_SOURCES = ['br', 'jp', 'unknown'] as const
export const footageSourceSchema = z.enum(FOOTAGE_SOURCES)
export type FootageSource = z.infer<typeof footageSourceSchema>

export const ROBOTS = ['A', 'B'] as const
export const robotSchema = z.enum(ROBOTS)
export type Robot = z.infer<typeof robotSchema>

const REVIEW_STATUSES = ['pending', 'approved', 'rejected'] as const
export const reviewStatusSchema = z.enum(REVIEW_STATUSES)
export type ReviewStatus = z.infer<typeof reviewStatusSchema>

export const EVENT_KINDS = ['round_start', 'first_contact', 'ring_out'] as const
const eventKindSchema = z.enum(EVENT_KINDS)
export type EventKind = z.infer<typeof eventKindSchema>

const unitInterval = z.number().min(0).max(1)
const positiveUnit = z.number().gt(0).max(1)

/** One robot box in YOLO convention: centre and size normalized to the frame. */
const boxSchema = z.object({
  cx: unitInterval,
  cy: unitInterval,
  w: positiveUnit,
  h: positiveUnit,
  robot: robotSchema.nullable(),
})
export type Box = z.infer<typeof boxSchema>

const frameIndexSchema = z.number().int().min(0)

const frameSchema = z.object({
  index: frameIndexSchema,
  hasImage: z.boolean(),
  boxes: z.array(boxSchema),
  review: reviewStatusSchema,
})
export type Frame = z.infer<typeof frameSchema>

export const clipEventSchema = z.object({
  kind: eventKindSchema,
  frameIndex: frameIndexSchema,
  robot: robotSchema.nullable(),
})
export type ClipEvent = z.infer<typeof clipEventSchema>

export const roundSchema = z
  .object({ start: frameIndexSchema, end: frameIndexSchema })
  .refine((round) => round.end >= round.start, { message: 'round ends before it starts' })
export type Round = z.infer<typeof roundSchema>

const clipSummarySchema = z.object({
  dataset: safeNameSchema,
  clip: safeNameSchema,
  role: roleSchema,
  source: footageSourceSchema,
  width: z.number().int(),
  height: z.number().int(),
  fps: z.number(),
  frameCount: z.number().int(),
  slotCount: z.number().int(),
  boxCount: z.number().int(),
  approvedCount: z.number().int(),
  rejectedCount: z.number().int(),
  bothRobotsCount: z.number().int(),
  identifiedCount: z.number().int(),
  eventCount: z.number().int(),
  roundCount: z.number().int(),
  hasVideo: z.boolean(),
  canExtractFrames: z.boolean(),
  thumbnailIndex: frameIndexSchema.nullable(),
})
export type ClipSummary = z.infer<typeof clipSummarySchema>

export const clipDetailSchema = clipSummarySchema.extend({
  stride: z.number().int().min(1),
  frames: z.array(frameSchema),
  events: z.array(clipEventSchema),
  rounds: z.array(roundSchema),
})
export type ClipDetail = z.infer<typeof clipDetailSchema>

const datasetSummarySchema = z.object({
  dataset: safeNameSchema,
  role: roleSchema,
  label: z.string(),
  description: z.string(),
  clips: z.array(clipSummarySchema),
})
export type DatasetSummary = z.infer<typeof datasetSummarySchema>

export const librarySchema = z.object({ datasets: z.array(datasetSummarySchema) })
export type Library = z.infer<typeof librarySchema>

/** Body of a frame save: the full box list replaces what is on disk for that frame. */
export const saveFrameSchema = z.object({
  boxes: z.array(boxSchema),
  review: reviewStatusSchema,
})
export type SaveFrame = z.infer<typeof saveFrameSchema>

export const saveTimelineSchema = z.object({
  events: z.array(clipEventSchema),
  rounds: z.array(roundSchema),
})
export type SaveTimeline = z.infer<typeof saveTimelineSchema>

export const moveClipSchema = z.object({ targetDataset: safeNameSchema })

/** Why each dataset can or cannot receive a clip, so the UI explains before the user tries. */
export const moveOptionsSchema = z.object({
  options: z.array(
    z.object({
      dataset: safeNameSchema,
      label: z.string(),
      role: roleSchema,
      blockers: z.array(z.string()),
    }),
  ),
})
export type MoveOptions = z.infer<typeof moveOptionsSchema>

export const clipLocationSchema = z.object({ dataset: safeNameSchema, clip: safeNameSchema })
