import { memo } from 'react'

import { cn } from '@/lib/utils'
import type { ClipEvent, EventKind, Frame, Round } from '@/shared/clip'
import { isIdentified } from '@/shared/identity'

const ROBOTS_PER_ROUND = 2
const SLOT_WIDTH = 0.9
const TRACK_HEIGHT = 7
const TRACK_GAP = 2
const TRACK_PITCH = TRACK_HEIGHT + TRACK_GAP

const TRACKS = ['Rounds', 'Eventos', 'Revisão', 'Caixas', 'A/B'] as const
const TIMELINE_HEIGHT = TRACKS.length * TRACK_PITCH - TRACK_GAP

const trackTop = (track: (typeof TRACKS)[number]) => TRACKS.indexOf(track) * TRACK_PITCH

const REVIEW_FILL: Record<Frame['review'], string> = {
  approved: 'fill-success',
  rejected: 'fill-destructive',
  pending: 'fill-foreground/25',
}

/** Display name and colour of each event kind, shared by the timeline and the inspector. */
export const EVENT_STYLE: Record<EventKind, { label: string; fill: string }> = {
  round_start: { label: 'Início', fill: 'fill-chart-2' },
  first_contact: { label: 'Contato', fill: 'fill-chart-3' },
  ring_out: { label: 'Ring-out', fill: 'fill-chart-5' },
}

function boxCountFill(frame: Frame): string {
  if (frame.boxes.length === ROBOTS_PER_ROUND) return 'fill-foreground/60'
  return frame.boxes.length === 0 ? 'fill-destructive/70' : 'fill-warning'
}

type TracksProps = { frames: Frame[]; events: ClipEvent[]; rounds: Round[] }

const Tracks = memo(function Tracks({ frames, events, rounds }: TracksProps) {
  const positionOf = new Map(frames.map((frame, position) => [frame.index, position]))
  const labeled = frames
    .map((frame, position) => ({ frame, position }))
    .filter(({ frame }) => frame.hasImage)
  return (
    <>
      {rounds.map((round) => {
        const start = positionOf.get(round.start)
        const end = positionOf.get(round.end)
        if (start === undefined || end === undefined) return null
        return (
          <rect
            key={`${round.start}:${round.end}`}
            x={start}
            y={trackTop('Rounds')}
            width={end - start + SLOT_WIDTH}
            height={TRACK_HEIGHT}
            rx="0.4"
            className="fill-foreground/35"
          />
        )
      })}
      {events
        .filter((event) => positionOf.has(event.frameIndex))
        .map((event) => (
          <rect
            key={`${event.kind}:${event.frameIndex}`}
            x={positionOf.get(event.frameIndex)}
            y={trackTop('Eventos')}
            width={SLOT_WIDTH}
            height={TRACK_HEIGHT}
            className={EVENT_STYLE[event.kind].fill}
          >
            <title>{EVENT_STYLE[event.kind].label}</title>
          </rect>
        ))}
      {labeled.map(({ frame, position }) => (
        <g key={frame.index}>
          <rect
            x={position}
            y={trackTop('Revisão')}
            width={SLOT_WIDTH}
            height={TRACK_HEIGHT}
            className={REVIEW_FILL[frame.review]}
          />
          <rect
            x={position}
            y={trackTop('Caixas')}
            width={SLOT_WIDTH}
            height={TRACK_HEIGHT}
            className={boxCountFill(frame)}
          />
          {isIdentified(frame.boxes) ? (
            <rect
              x={position}
              y={trackTop('A/B')}
              width={SLOT_WIDTH}
              height={TRACK_HEIGHT}
              className="fill-robot-a"
            />
          ) : null}
        </g>
      ))}
    </>
  )
})

type TimelineProps = TracksProps & {
  position: number
  openRoundStart: number | null
  onSeek: (position: number) => void
}

/**
 * The whole clip at a glance: one column per frame slot, one row per kind of annotation.
 * Gaps and colour breaks show where work is left. Click or drag to move the cursor.
 */
export function Timeline({ position, openRoundStart, onSeek, ...tracks }: TimelineProps) {
  const slotCount = tracks.frames.length
  const openRoundPosition = tracks.frames.findIndex((frame) => frame.index === openRoundStart)

  return (
    <div className="flex min-h-0 flex-1 gap-3 px-3.5 pb-3.5">
      <ul className="text-muted-foreground flex w-14 flex-none flex-col justify-between font-mono text-[11px] leading-none">
        {TRACKS.map((track) => (
          <li key={track} className="flex flex-1 items-center">
            {track}
          </li>
        ))}
      </ul>
      <div className="relative min-h-0 flex-1">
        <input
          type="range"
          aria-label="Posição no clip"
          aria-valuetext={`quadro ${position + 1} de ${slotCount}`}
          min={0}
          max={slotCount - 1}
          step={1}
          value={position}
          onChange={(event) => onSeek(event.target.valueAsNumber)}
          className="peer absolute inset-0 z-10 size-full cursor-pointer appearance-none opacity-0 [&::-moz-range-thumb]:w-px [&::-webkit-slider-thumb]:w-px [&::-webkit-slider-thumb]:appearance-none"
        />
        <svg
          aria-hidden="true"
          viewBox={`0 0 ${slotCount} ${TIMELINE_HEIGHT}`}
          preserveAspectRatio="none"
          className="peer-focus-visible:ring-ring/60 size-full rounded-sm peer-focus-visible:ring-2"
        >
          <Tracks {...tracks} />
          {openRoundPosition === -1 ? null : (
            <rect
              x={openRoundPosition}
              y={trackTop('Rounds')}
              width={SLOT_WIDTH}
              height={TRACK_HEIGHT}
              className="fill-warning"
            />
          )}
          <rect
            x={position}
            y="0"
            width={SLOT_WIDTH}
            height={TIMELINE_HEIGHT}
            className={cn('fill-foreground/15 stroke-foreground')}
            strokeWidth="1"
            vectorEffect="non-scaling-stroke"
          />
        </svg>
      </div>
    </div>
  )
}
