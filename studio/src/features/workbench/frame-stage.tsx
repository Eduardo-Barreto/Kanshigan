import { useMemo, useRef, useState, type PointerEvent } from 'react'

import { cn } from '@/lib/utils'
import { cornersOf, type Point } from '@/shared/box-editing'
import type { Box } from '@/shared/clip'

import { beginDrag, continueDrag, endDrag, type Drag } from './stage-drag'

const CORNER_GRAB_PX = 10
const PERCENT = 100
const SELECTED_STROKE_PX = 2.5
const IDLE_STROKE_PX = 1.5

const STROKE_BY_ROBOT = {
  A: 'stroke-robot-a',
  B: 'stroke-robot-b',
  none: 'stroke-[oklch(0.97_0_0)]',
}
const FILL_BY_ROBOT = { A: 'fill-robot-a', B: 'fill-robot-b', none: 'fill-[oklch(0.97_0_0)]' }

const percent = (fraction: number) => `${fraction * PERCENT}%`

type FrameStageProps = {
  imageUrl: string
  imageLabel: string
  width: number
  height: number
  boxes: Box[]
  selected: number | null
  onEdit: (boxes: Box[], selected: number | null) => void
  onSelect: (selected: number | null) => void
}

function BoxOutline({ box, isSelected }: { box: Box; isSelected: boolean }) {
  const tone = box.robot ?? 'none'
  const [topLeft] = cornersOf(box)
  return (
    <g>
      <rect
        x={percent(topLeft.x)}
        y={percent(topLeft.y)}
        width={percent(box.w)}
        height={percent(box.h)}
        className={cn('fill-transparent', STROKE_BY_ROBOT[tone])}
        strokeWidth={isSelected ? SELECTED_STROKE_PX : IDLE_STROKE_PX}
      />
      {box.robot ? (
        <text
          x={percent(topLeft.x)}
          y={percent(topLeft.y)}
          dy="-5"
          className={cn(
            'font-mono text-xs font-semibold [paint-order:stroke] stroke-black/70',
            FILL_BY_ROBOT[tone],
          )}
          strokeWidth="3"
        >
          {box.robot}
        </text>
      ) : null}
      {isSelected
        ? cornersOf(box).map((corner) => (
            <circle
              key={`${corner.x}:${corner.y}`}
              cx={percent(corner.x)}
              cy={percent(corner.y)}
              r="4.5"
              className={cn('stroke-black/60', FILL_BY_ROBOT[tone])}
            />
          ))
        : null}
    </g>
  )
}

function locate(event: PointerEvent<SVGSVGElement>): { point: Point; tolerance: Point } {
  const bounds = event.currentTarget.getBoundingClientRect()
  return {
    point: {
      x: (event.clientX - bounds.left) / bounds.width,
      y: (event.clientY - bounds.top) / bounds.height,
    },
    tolerance: { x: CORNER_GRAB_PX / bounds.width, y: CORNER_GRAB_PX / bounds.height },
  }
}

/** The frame image with its boxes drawn on top, editable by pointer: move, resize by a corner, or draw. */
export function FrameStage(props: FrameStageProps) {
  const { boxes, selected, onEdit, onSelect } = props
  const drag = useRef<Drag | null>(null)
  const [preview, setPreview] = useState<Box | null>(null)
  const fitStyle = useMemo(
    () => ({
      aspectRatio: `${props.width} / ${props.height}`,
      width: `min(100cqw, calc(100cqh * ${props.width} / ${props.height}))`,
    }),
    [props.width, props.height],
  )

  const handlePointerDown = (event: PointerEvent<SVGSVGElement>) => {
    if (event.button !== 0) return
    const { point, tolerance } = locate(event)
    event.currentTarget.setPointerCapture(event.pointerId)
    drag.current = beginDrag(boxes, selected, point, tolerance)
    onSelect(drag.current.kind === 'draw' ? null : drag.current.boxIndex)
  }

  const handlePointerMove = (event: PointerEvent<SVGSVGElement>) => {
    if (drag.current === null) return
    const result = continueDrag(drag.current, boxes, locate(event).point)
    setPreview(result.preview)
    if (drag.current.kind !== 'draw') onEdit(result.boxes, result.selected)
  }

  const handlePointerUp = (event: PointerEvent<SVGSVGElement>) => {
    if (drag.current === null) return
    const finished = drag.current
    drag.current = null
    setPreview(null)
    const result = endDrag(finished, boxes, locate(event).point)
    if (result.boxes !== boxes) onEdit(result.boxes, result.selected)
  }

  return (
    <div className="flex min-h-0 flex-1 items-center justify-center [container-type:size]">
      <div className="relative overflow-hidden rounded-md bg-black" style={fitStyle}>
        <img
          src={props.imageUrl}
          alt={props.imageLabel}
          draggable={false}
          className="size-full select-none"
        />
        <svg
          role="application"
          aria-label="Caixas do quadro. Arraste para mover, puxe um canto para redimensionar, arraste no vazio para criar."
          className="absolute inset-0 size-full cursor-crosshair touch-none"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
        >
          {boxes.map((box, position) => (
            <BoxOutline
              key={`${position}:${box.robot ?? 'none'}`}
              box={box}
              isSelected={position === selected}
            />
          ))}
          {preview ? (
            <rect
              x={percent(preview.cx - preview.w / 2)}
              y={percent(preview.cy - preview.h / 2)}
              width={percent(preview.w)}
              height={percent(preview.h)}
              className="fill-white/10 stroke-white"
              strokeDasharray="5 4"
              strokeWidth="1.5"
            />
          ) : null}
        </svg>
      </div>
    </div>
  )
}
