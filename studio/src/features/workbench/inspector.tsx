import { Check, Trash2, X } from 'lucide-react'
import type { ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import { Kbd } from '@/components/window'
import { formatClock } from '@/lib/format'
import { cn } from '@/lib/utils'
import { EVENT_KINDS, ROBOTS, type Box, type ClipDetail, type ReviewStatus } from '@/shared/clip'

import { EVENT_STYLE } from './timeline'
import {
  currentBoxes,
  currentFrame,
  timeOfSlot,
  type WorkbenchSnapshot,
  type WorkbenchStore,
} from './workbench-store'

const HALF_SPEED = 0.5
const DOUBLE_SPEED = 2
const QUADRUPLE_SPEED = 4
const SPEEDS = [HALF_SPEED, 1, DOUBLE_SPEED, QUADRUPLE_SPEED] as const
const EVENT_KEYS = { round_start: 'H', first_contact: 'T', ring_out: 'O' } as const
const REVIEW_LABEL: Record<ReviewStatus, string> = {
  approved: 'Aprovado',
  rejected: 'Rejeitado',
  pending: 'Pendente',
}
const REVIEW_TONE: Record<ReviewStatus, string> = {
  approved: 'text-success',
  rejected: 'text-destructive',
  pending: 'text-foreground',
}
const SWATCH_BY_ROBOT = { A: 'bg-robot-a', B: 'bg-robot-b', none: 'bg-[oklch(0.97_0_0)]' }

type PanelProps = { snapshot: WorkbenchSnapshot; store: WorkbenchStore }

function Section({
  title,
  aside,
  children,
}: {
  title: string
  aside?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="border-border flex flex-col gap-2.5 border-b px-4 py-3.5 last:border-b-0">
      <h3 className="text-muted-foreground flex items-center justify-between text-xs font-medium">
        {title}
        {aside}
      </h3>
      {children}
    </section>
  )
}

function ReviewPanel({ snapshot, store }: PanelProps) {
  const frame = currentFrame(snapshot)
  return (
    <Section title="Revisão do quadro">
      <p
        className={cn(
          'tabular text-[22px] leading-7 font-semibold tracking-[-0.02em]',
          REVIEW_TONE[frame.review],
        )}
      >
        {REVIEW_LABEL[frame.review]}
      </p>
      <div className="grid grid-cols-2 gap-2">
        <Button variant="outline" onClick={() => store.review('approved')}>
          <Check /> Aprovar <Kbd>Enter</Kbd>
        </Button>
        <Button variant="outline" onClick={() => store.review('rejected')}>
          <X /> Rejeitar <Kbd>X</Kbd>
        </Button>
      </div>
    </Section>
  )
}

function BoxRow({ box, position, snapshot, store }: PanelProps & { box: Box; position: number }) {
  const { detail, selected } = snapshot
  const size = `${Math.round(box.w * detail.width)}×${Math.round(box.h * detail.height)}`
  return (
    <li
      className={cn(
        'flex h-9 items-center gap-2 rounded-md pr-1 pl-2',
        position === selected && 'bg-muted/70',
      )}
    >
      <button
        type="button"
        className="flex min-w-0 flex-1 items-center gap-2 text-left outline-none"
        onClick={() => store.select(position)}
      >
        <span
          className={cn('size-2 flex-none rounded-[2px]', SWATCH_BY_ROBOT[box.robot ?? 'none'])}
        />
        <span className="truncate text-[13px]">
          {box.robot ? `Robô ${box.robot}` : 'Sem identidade'}
        </span>
        <span className="text-muted-foreground tabular text-[11px]">{size}</span>
      </button>
      {ROBOTS.map((robot) => (
        <Button
          key={robot}
          size="icon-xs"
          variant={box.robot === robot ? 'secondary' : 'ghost'}
          aria-pressed={box.robot === robot}
          aria-label={`Marcar caixa ${position + 1} como robô ${robot}`}
          onClick={() => {
            store.select(position)
            store.assign(robot)
          }}
        >
          {robot}
        </Button>
      ))}
      <Button
        size="icon-xs"
        variant="ghost"
        aria-label={`Apagar caixa ${position + 1}`}
        onClick={() => {
          store.select(position)
          store.deleteSelected()
        }}
      >
        <Trash2 />
      </Button>
    </li>
  )
}

function BoxesPanel({ snapshot, store }: PanelProps) {
  const boxes = currentBoxes(snapshot)
  return (
    <Section title="Caixas" aside={<span className="tabular">{boxes.length}</span>}>
      {boxes.length === 0 ? (
        <p className="text-muted-foreground text-xs">
          Nenhuma caixa. Arraste sobre o quadro para criar uma.
        </p>
      ) : (
        <ul className="-mx-2 flex flex-col gap-px">
          {boxes.map((box, position) => (
            <BoxRow
              key={`${position}:${box.robot ?? 'none'}`}
              box={box}
              position={position}
              snapshot={snapshot}
              store={store}
            />
          ))}
        </ul>
      )}
      <div className="grid grid-cols-2 gap-2">
        <Button variant="outline" size="sm" onClick={() => store.swapIdentities()}>
          Trocar A e B <Kbd>Q</Kbd>
        </Button>
        <Button variant="outline" size="sm" onClick={() => store.propagateForward()}>
          Propagar <Kbd>I</Kbd>
        </Button>
      </div>
    </Section>
  )
}

function MarkList({ detail, store }: { detail: ClipDetail; store: WorkbenchStore }) {
  if (detail.events.length === 0 && detail.rounds.length === 0) return null
  return (
    <ul className="-mx-2 flex flex-col gap-px text-[13px]">
      {detail.rounds.map((round, position) => (
        <li key={`${round.start}:${round.end}`} className="flex h-8 items-center gap-2 pr-1 pl-2">
          <span className="flex-1">Round {position + 1}</span>
          <span className="text-muted-foreground tabular text-[11px]">
            {formatClock(timeOfSlot(detail, round.start))} a{' '}
            {formatClock(timeOfSlot(detail, round.end))}
          </span>
          <Button
            size="icon-xs"
            variant="ghost"
            aria-label={`Remover round ${position + 1}`}
            onClick={() => store.removeRound(position)}
          >
            <Trash2 />
          </Button>
        </li>
      ))}
      {detail.events.map((event, position) => (
        <li
          key={`${event.kind}:${event.frameIndex}`}
          className="flex h-8 items-center gap-2 pr-1 pl-2"
        >
          <span className="flex-1">
            {EVENT_STYLE[event.kind].label}
            {event.robot ? ` do robô ${event.robot}` : ''}
          </span>
          <span className="text-muted-foreground tabular text-[11px]">
            {formatClock(timeOfSlot(detail, event.frameIndex))}
          </span>
          <Button
            size="icon-xs"
            variant="ghost"
            aria-label={`Remover ${EVENT_STYLE[event.kind].label} em ${formatClock(timeOfSlot(detail, event.frameIndex))}`}
            onClick={() => store.removeEvent(position)}
          >
            <Trash2 />
          </Button>
        </li>
      ))}
    </ul>
  )
}

function MarksPanel({ snapshot, store }: PanelProps) {
  const isRoundOpen = snapshot.openRoundStart !== null
  return (
    <Section title="Marcações neste quadro">
      <div className="grid grid-cols-3 gap-2">
        {EVENT_KINDS.map((kind) => (
          <Button key={kind} variant="outline" size="sm" onClick={() => store.toggleEvent(kind)}>
            {EVENT_STYLE[kind].label} <Kbd>{EVENT_KEYS[kind]}</Kbd>
          </Button>
        ))}
      </div>
      <Button
        variant={isRoundOpen ? 'secondary' : 'outline'}
        size="sm"
        onClick={() => store.markRound()}
      >
        {isRoundOpen ? 'Fechar round aqui' : 'Abrir round aqui'} <Kbd>R</Kbd>
      </Button>
      <MarkList detail={snapshot.detail} store={store} />
    </Section>
  )
}

function PlaybackPanel({ snapshot, store }: PanelProps) {
  return (
    <Section title="Reprodução">
      <fieldset className="grid grid-cols-4 gap-1">
        <legend className="sr-only">Velocidade</legend>
        {SPEEDS.map((speed) => (
          <Button
            key={speed}
            size="sm"
            variant={snapshot.speed === speed ? 'secondary' : 'ghost'}
            aria-pressed={snapshot.speed === speed}
            onClick={() => store.setSpeed(speed)}
          >
            {speed.toLocaleString('pt-BR')}x
          </Button>
        ))}
      </fieldset>
      <Label className="flex items-center gap-2 text-[13px] font-normal">
        <Checkbox
          checked={snapshot.options.autoApprove}
          onCheckedChange={(checked) => store.setOption('autoApprove', checked === true)}
        />
        Aprovar ao passar pelo quadro no play
      </Label>
      <Label className="flex items-center gap-2 text-[13px] font-normal">
        <Checkbox
          checked={snapshot.options.propagate}
          onCheckedChange={(checked) => store.setOption('propagate', checked === true)}
        />
        Propor A e B a partir do quadro anterior
      </Label>
    </Section>
  )
}

/** Side panel of the workbench: review, boxes with identity, timeline marks and playback switches. */
export function Inspector({ snapshot, store }: PanelProps) {
  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <ReviewPanel snapshot={snapshot} store={store} />
      <BoxesPanel snapshot={snapshot} store={store} />
      <MarksPanel snapshot={snapshot} store={store} />
      <PlaybackPanel snapshot={snapshot} store={store} />
    </div>
  )
}
