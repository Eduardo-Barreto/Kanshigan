import { useSuspenseQuery } from '@tanstack/react-query'
import { Clapperboard, Film, ListChecks, Scissors, Sparkles } from 'lucide-react'
import { useMemo, useState } from 'react'

import { Chip, Window, WindowTitleBar } from '@/components/window'
import { jobsQuery, libraryQuery, videosQuery } from '@/lib/api'
import type { Job, RawVideo } from '@/shared/ingest'

import { AnnotatePanel } from './annotate-panel'
import { JobList } from './job-list'
import { RoundPanel, type EditableWindow } from './round-panel'
import { VideoPanel, videoKey } from './video-panel'

function segmentationOf(jobs: Job[], video: RawVideo): Job | undefined {
  return jobs.find(
    (job) =>
      job.kind === 'segment' && job.video !== null && videoKey(job.video) === videoKey(video),
  )
}

function EmptySelection() {
  return (
    <div className="text-muted-foreground flex flex-1 flex-col items-center justify-center gap-1.5 text-center">
      <Film className="size-5" />
      <p className="text-foreground text-[15px] font-semibold">Escolha um vídeo</p>
      <p className="max-w-80 text-[13px]">
        Envie um arquivo, baixe do YouTube ou escolha um vídeo bruto na lista para cortar em rounds.
      </p>
    </div>
  )
}

/** Brings a new video into the library: pick it, settle the rounds, then cut and annotate. */
export function IngestScreen() {
  const { data: videos } = useSuspenseQuery(videosQuery)
  const { data: jobs } = useSuspenseQuery(jobsQuery)
  const { data: library } = useSuspenseQuery(libraryQuery)
  const [selectedKey, setSelectedKey] = useState<string | null>(null)
  const [draftWindows, setDraftWindows] = useState<EditableWindow[] | null>(null)

  const video = videos.find((candidate) => videoKey(candidate) === selectedKey)
  const segmentation = video ? segmentationOf(jobs, video) : undefined
  const detected = segmentation?.roundWindows
  const windows = useMemo(
    () => draftWindows ?? (detected ?? []).map((window) => ({ ...window, included: true })),
    [draftWindows, detected],
  )
  const included = useMemo(
    () => windows.filter((window) => window.included && window.end > window.start),
    [windows],
  )
  const candidateDatasets = useMemo(
    () =>
      library.datasets
        .filter((dataset) => dataset.role === 'gold_candidate')
        .map((dataset) => dataset.dataset),
    [library],
  )

  return (
    <div className="grid h-full grid-cols-[320px_minmax(0,1fr)_360px] grid-rows-[minmax(0,1fr)_minmax(0,0.7fr)] gap-3.5">
      <Window aria-label="Vídeos brutos" className="row-span-2">
        <WindowTitleBar icon={<Clapperboard />} title="Vídeos brutos">
          <Chip label="vídeos" value={videos.length} />
        </WindowTitleBar>
        <VideoPanel
          videos={videos}
          selectedKey={selectedKey}
          onSelect={(selected) => {
            setSelectedKey(videoKey(selected))
            setDraftWindows(null)
          }}
        />
      </Window>
      <Window aria-label="Rounds" className="row-span-2">
        <WindowTitleBar icon={<Scissors />} title="Rounds" subtitle={video?.name}>
          <Chip label="marcados" value={windows.length} />
          <Chip label="incluídos" value={included.length} />
        </WindowTitleBar>
        {video ? (
          <RoundPanel
            key={videoKey(video)}
            video={video}
            windows={windows}
            isDetecting={segmentation?.status === 'queued' || segmentation?.status === 'running'}
            onChange={setDraftWindows}
            onDetect={() => setDraftWindows(null)}
          />
        ) : (
          <EmptySelection />
        )}
      </Window>
      <Window aria-label="Anotar com SAM 3">
        <WindowTitleBar icon={<Sparkles />} title="Cortar e anotar" subtitle="SAM 3" />
        {video ? (
          <AnnotatePanel
            key={videoKey(video)}
            video={video}
            windows={included}
            candidateDatasets={candidateDatasets}
          />
        ) : (
          <p className="text-muted-foreground px-3.5 text-xs">
            Escolha um vídeo para liberar o corte.
          </p>
        )}
      </Window>
      <Window aria-label="Tarefas">
        <WindowTitleBar
          icon={<ListChecks />}
          title="Tarefas"
          subtitle="uma por vez, por causa da GPU"
        >
          <Chip
            label="ativas"
            value={jobs.filter((job) => job.status === 'queued' || job.status === 'running').length}
          />
        </WindowTitleBar>
        <JobList jobs={jobs} />
      </Window>
    </div>
  )
}
