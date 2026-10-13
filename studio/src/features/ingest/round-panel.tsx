import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Play, Plus, ScanSearch, Trash2 } from 'lucide-react'
import { useRef, useState } from 'react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { queryKeys, rawVideoUrl, startSegmentation } from '@/lib/api'
import type { RawVideo, RoundWindow } from '@/shared/ingest'

const TIME_STEP_SECONDS = 0.1
const TIME_DECIMALS = 1

/** A round window the annotator can still adjust or leave out of the job. */
export type EditableWindow = RoundWindow & { included: boolean }

type RoundPanelProps = {
  video: RawVideo
  windows: EditableWindow[]
  isDetecting: boolean
  onChange: (windows: EditableWindow[]) => void
  onDetect: () => void
}

type WindowRowProps = {
  window: EditableWindow
  position: number
  onEdit: (patch: Partial<EditableWindow>) => void
  onRemove: () => void
  onPreview: () => void
}

function WindowRow({ window, position, onEdit, onRemove, onPreview }: WindowRowProps) {
  const label = `round ${position + 1}`
  return (
    <li className="flex h-10 items-center gap-2 px-2">
      <Checkbox
        aria-label={`Incluir ${label}`}
        checked={window.included}
        onCheckedChange={(checked) => onEdit({ included: checked === true })}
      />
      <span className="w-16 text-[13px]">Round {position + 1}</span>
      <Input
        type="number"
        aria-label={`Início do ${label}, em segundos`}
        className="tabular h-7 w-24"
        step={TIME_STEP_SECONDS}
        min={0}
        value={window.start}
        onChange={(event) => onEdit({ start: event.target.valueAsNumber })}
      />
      <span className="text-muted-foreground text-xs">a</span>
      <Input
        type="number"
        aria-label={`Fim do ${label}, em segundos`}
        className="tabular h-7 w-24"
        step={TIME_STEP_SECONDS}
        min={0}
        value={window.end}
        onChange={(event) => onEdit({ end: event.target.valueAsNumber })}
      />
      <span className="text-muted-foreground tabular flex-1 text-[11px]">
        {(window.end - window.start).toFixed(TIME_DECIMALS)} s
      </span>
      <Button
        size="icon-xs"
        variant="ghost"
        aria-label={`Ver o ${label} no vídeo`}
        onClick={onPreview}
      >
        <Play />
      </Button>
      <Button size="icon-xs" variant="ghost" aria-label={`Remover o ${label}`} onClick={onRemove}>
        <Trash2 />
      </Button>
    </li>
  )
}

function useDetectRounds(video: RawVideo) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () => startSegmentation(video),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: queryKeys.jobs }),
    onError: (error) => toast.error(error.message),
  })
}

/** Middle window of the ingest screen: watch the raw video and settle where each round starts and ends. */
export function RoundPanel({ video, windows, isDetecting, onChange, onDetect }: RoundPanelProps) {
  const player = useRef<HTMLVideoElement>(null)
  const [openStart, setOpenStart] = useState<number | null>(null)
  const detect = useDetectRounds(video)

  const currentTime = () => Number((player.current?.currentTime ?? 0).toFixed(TIME_DECIMALS))
  const editAt = (position: number, patch: Partial<EditableWindow>) =>
    onChange(
      windows.map((window, index) => (index === position ? { ...window, ...patch } : window)),
    )
  const markHere = () => {
    if (openStart === null) {
      setOpenStart(currentTime())
      return
    }
    const [start, end] = [openStart, currentTime()].toSorted((first, second) => first - second)
    setOpenStart(null)
    if (start !== undefined && end !== undefined && end > start) {
      onChange([...windows, { start, end, included: true }])
    }
  }
  const preview = (window: EditableWindow) => {
    if (player.current === null) return
    player.current.currentTime = window.start
    void player.current.play()
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 px-3.5 pb-3.5">
      <video
        ref={player}
        src={rawVideoUrl(video)}
        controls
        muted
        className="min-h-0 flex-1 rounded-md bg-black"
      >
        <track kind="captions" />
      </video>
      <div className="flex flex-none items-center gap-2">
        <Button
          variant="outline"
          disabled={detect.isPending || isDetecting}
          onClick={() => {
            onDetect()
            detect.mutate()
          }}
        >
          <ScanSearch /> {isDetecting ? 'Detectando…' : 'Detectar rounds'}
        </Button>
        <Button variant={openStart === null ? 'outline' : 'secondary'} onClick={markHere}>
          <Plus />
          {openStart === null
            ? 'Abrir round no tempo atual'
            : `Fechar round aberto em ${openStart} s`}
        </Button>
      </div>
      {windows.length === 0 ? (
        <p className="text-muted-foreground flex-none text-xs">
          Nenhum round ainda. Detecte automaticamente ou marque o início e o fim no vídeo.
        </p>
      ) : (
        <ul className="-mx-2 max-h-[40%] flex-none overflow-y-auto">
          {windows.map((window, position) => (
            <WindowRow
              key={`round-${position}`}
              window={window}
              position={position}
              onEdit={(patch) => editAt(position, patch)}
              onRemove={() => onChange(windows.filter((_window, index) => index !== position))}
              onPreview={() => preview(window)}
            />
          ))}
        </ul>
      )}
    </div>
  )
}
