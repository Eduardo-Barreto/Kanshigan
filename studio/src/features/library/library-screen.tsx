import { useSuspenseQuery } from '@tanstack/react-query'
import { Award, FlaskConical, GraduationCap, Hourglass } from 'lucide-react'
import type { ReactNode } from 'react'

import { Chip, Window, WindowTitleBar } from '@/components/window'
import { libraryQuery } from '@/lib/api'
import { formatCount, formatShare } from '@/lib/format'
import type { ClipSummary, DatasetSummary, Role } from '@/shared/clip'

import { ClipRow } from './clip-row'

function clipsOf(datasets: DatasetSummary[]): ClipSummary[] {
  return datasets.flatMap((dataset) => dataset.clips)
}

function sum(clips: ClipSummary[], pick: (clip: ClipSummary) => number): number {
  return clips.reduce((total, clip) => total + pick(clip), 0)
}

function Readout({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="border-border grid gap-1 border-l px-4 py-2.5 first:border-l-0">
      <dt className="text-muted-foreground text-xs whitespace-nowrap">{label}</dt>
      <dd className="tabular text-[26px] leading-8 font-medium tracking-[-0.02em]">{value}</dd>
      {note ? <dd className="text-muted-foreground text-[11px]">{note}</dd> : null}
    </div>
  )
}

function GoldReadouts({ clips }: { clips: ClipSummary[] }) {
  const frames = sum(clips, (clip) => clip.frameCount - clip.rejectedCount)
  return (
    <dl className="grid flex-none grid-cols-5 px-2 pb-3">
      <Readout label="Quadros" value={formatCount(frames)} note={`${clips.length} clips`} />
      <Readout
        label="Com os 2 robôs"
        value={formatShare(
          sum(clips, (clip) => clip.bothRobotsCount),
          frames,
        )}
        note={`${formatCount(sum(clips, (clip) => clip.bothRobotsCount))} quadros`}
      />
      <Readout
        label="Identidade A/B"
        value={formatShare(
          sum(clips, (clip) => clip.identifiedCount),
          frames,
        )}
        note={`${formatCount(sum(clips, (clip) => clip.identifiedCount))} quadros`}
      />
      <Readout label="Eventos" value={formatCount(sum(clips, (clip) => clip.eventCount))} />
      <Readout label="Rounds" value={formatCount(sum(clips, (clip) => clip.roundCount))} />
    </dl>
  )
}

function DatasetGroup({ dataset }: { dataset: DatasetSummary }) {
  return (
    <li>
      <div className="flex items-baseline gap-2 px-2 pt-2 pb-1">
        <h3 className="text-xs font-medium">{dataset.label}</h3>
        <p className="text-muted-foreground truncate text-xs">{dataset.description}</p>
      </div>
      <ul className="flex flex-col gap-px">
        {dataset.clips.map((clip) => (
          <li key={clip.clip}>
            <ClipRow clip={clip} />
          </li>
        ))}
      </ul>
    </li>
  )
}

type LibraryWindowProps = {
  icon: ReactNode
  title: string
  subtitle: string
  datasets: DatasetSummary[]
  className?: string
  children?: ReactNode
}

function LibraryWindow({
  icon,
  title,
  subtitle,
  datasets,
  className,
  children,
}: LibraryWindowProps) {
  const clips = clipsOf(datasets)
  return (
    <Window className={className} aria-label={title}>
      <WindowTitleBar icon={icon} title={title} subtitle={subtitle}>
        <Chip label="clips" value={formatCount(clips.length)} />
        <Chip label="quadros" value={formatCount(sum(clips, (clip) => clip.frameCount))} />
      </WindowTitleBar>
      {children}
      <ul className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
        {datasets.map((dataset) => (
          <DatasetGroup key={dataset.dataset} dataset={dataset} />
        ))}
      </ul>
    </Window>
  )
}

/** The library: every dataset tiled by role, with the gold test set as the largest surface. */
export function LibraryScreen() {
  const { data: library } = useSuspenseQuery(libraryQuery)
  const withRole = (...roles: Role[]) =>
    library.datasets.filter((dataset) => roles.includes(dataset.role))
  const gold = withRole('gold')
  return (
    <div className="grid h-full grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)] grid-rows-[minmax(0,1fr)_minmax(0,1fr)] gap-3.5">
      <LibraryWindow icon={<Award />} title="Gold" subtitle="conjunto de teste" datasets={gold}>
        <GoldReadouts clips={clipsOf(gold)} />
      </LibraryWindow>
      <LibraryWindow
        icon={<Hourglass />}
        title="Candidatos a gold"
        subtitle="esperando revisão"
        datasets={withRole('gold_candidate')}
      />
      <LibraryWindow
        icon={<GraduationCap />}
        title="Treino e validação"
        subtitle="o que o detector vê"
        datasets={withRole('train', 'val')}
      />
      <LibraryWindow
        icon={<FlaskConical />}
        title="Experimentos"
        subtitle="fora do treino e da avaliação"
        datasets={withRole('other')}
      />
    </div>
  )
}
