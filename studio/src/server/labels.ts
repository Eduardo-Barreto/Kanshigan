import { z } from 'zod'

const COORDINATE_DECIMALS = 6
const ROBOT_CLASS_ID = 0
const FIELDS_PER_LINE = 5

const labelBoxSchema = z.object({
  cx: z.number().min(0).max(1),
  cy: z.number().min(0).max(1),
  w: z.number().gt(0).max(1),
  h: z.number().gt(0).max(1),
})

/** Box geometry as a YOLO label line stores it: no identity, one implicit class. */
export type LabelBox = z.infer<typeof labelBoxSchema>

function parseLine(line: string): LabelBox | null {
  const fields = line.trim().split(/\s+/)
  if (fields.length < FIELDS_PER_LINE) return null
  const [cx, cy, w, h] = fields.slice(1, FIELDS_PER_LINE).map(Number)
  const parsed = labelBoxSchema.safeParse({ cx, cy, w, h })
  return parsed.success ? parsed.data : null
}

/** Parses a YOLO label file, dropping blank and malformed lines instead of failing the frame. */
export function parseLabels(text: string): LabelBox[] {
  return text
    .split('\n')
    .map(parseLine)
    .filter((box) => box !== null)
}

/** Serializes boxes to YOLO label text, one `class cx cy w h` line per box. */
export function serializeLabels(boxes: LabelBox[]): string {
  return boxes
    .map((box) =>
      [ROBOT_CLASS_ID, box.cx, box.cy, box.w, box.h]
        .map((field, position) => (position === 0 ? field : field.toFixed(COORDINATE_DECIMALS)))
        .join(' '),
    )
    .map((line) => `${line}\n`)
    .join('')
}
