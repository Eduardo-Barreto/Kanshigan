import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Download, Upload } from 'lucide-react'
import { useState, type ChangeEvent, type FormEvent } from 'react'
import { toast } from 'sonner'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { queryKeys, startDownload, uploadVideo } from '@/lib/api'
import { formatClock } from '@/lib/format'
import { cn } from '@/lib/utils'
import { safeNameSchema } from '@/shared/clip'
import { downloadRequestSchema, type RawSource, type RawVideo } from '@/shared/ingest'

const MS_PER_SECOND = 1000
const SOURCES: { value: RawSource; label: string }[] = [
  { value: 'br', label: 'BR, câmera de mão' },
  { value: 'jp', label: 'JP, câmera fixa' },
]

/** Derives a directory-safe video name from an uploaded file name. */
export function videoNameFromFile(fileName: string): string {
  const stem = fileName.replace(/\.[^.]+$/, '')
  const cleaned = stem.replace(/[^A-Za-z0-9_-]+/g, '_').replace(/^[^A-Za-z0-9]+/, '')
  return safeNameSchema.safeParse(cleaned).success ? cleaned : 'video'
}

/** Stable identifier of a raw video inside the ingest screen. */
export function videoKey(video: { source: RawSource; name: string }): string {
  return `${video.source}/${video.name}`
}

function useAddVideo(source: RawSource) {
  const queryClient = useQueryClient()
  const upload = useMutation({
    mutationFn: (file: File) => uploadVideo({ source, name: videoNameFromFile(file.name) }, file),
    onSuccess: (video) => {
      toast.success(`${video.name} enviado`)
      return queryClient.invalidateQueries({ queryKey: queryKeys.videos })
    },
    onError: (error) => toast.error(error.message),
  })
  const download = useMutation({
    mutationFn: (url: string) => startDownload({ url, source }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: queryKeys.jobs }),
    onError: (error) => toast.error(error.message),
  })
  return { upload, download }
}

function AddVideo() {
  const [source, setSource] = useState<RawSource>('br')
  const [url, setUrl] = useState('')
  const { upload, download } = useAddVideo(source)

  const handleFile = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) upload.mutate(file)
    event.target.value = ''
  }
  const handleDownload = (event: FormEvent) => {
    event.preventDefault()
    const request = downloadRequestSchema.safeParse({ url, source })
    if (!request.success) {
      toast.error('Cole um link completo do YouTube')
      return
    }
    download.mutate(request.data.url, { onSuccess: () => setUrl('') })
  }

  return (
    <div className="border-border flex flex-none flex-col gap-2.5 border-b px-3.5 pb-3.5">
      <fieldset className="grid grid-cols-2 gap-1">
        <legend className="sr-only">Fonte do vídeo novo</legend>
        {SOURCES.map((option) => (
          <Button
            key={option.value}
            size="sm"
            variant={source === option.value ? 'secondary' : 'ghost'}
            aria-pressed={source === option.value}
            onClick={() => setSource(option.value)}
          >
            {option.label}
          </Button>
        ))}
      </fieldset>
      <Button asChild variant="outline" disabled={upload.isPending}>
        <label>
          <Upload />
          {upload.isPending ? 'Enviando…' : 'Enviar arquivo de vídeo'}
          <input type="file" accept="video/*" className="sr-only" onChange={handleFile} />
        </label>
      </Button>
      <form className="flex gap-2" onSubmit={handleDownload}>
        <Input
          aria-label="Link do YouTube"
          placeholder="https://youtu.be/…"
          value={url}
          onChange={(event) => setUrl(event.target.value)}
        />
        <Button type="submit" variant="outline" disabled={download.isPending}>
          <Download /> Baixar
        </Button>
      </form>
    </div>
  )
}

type VideoPanelProps = {
  videos: RawVideo[]
  selectedKey: string | null
  onSelect: (video: RawVideo) => void
}

/** Left window of the ingest screen: add a raw video and pick which one to cut. */
export function VideoPanel({ videos, selectedKey, onSelect }: VideoPanelProps) {
  return (
    <>
      <AddVideo />
      <ul className="min-h-0 flex-1 overflow-y-auto p-2">
        {videos.map((video) => (
          <li key={videoKey(video)}>
            <button
              type="button"
              aria-pressed={videoKey(video) === selectedKey}
              onClick={() => onSelect(video)}
              className={cn(
                'hover:bg-muted/50 focus-visible:ring-ring/60 flex h-11 w-full items-center gap-2 rounded-md px-2 text-left outline-none focus-visible:ring-2',
                videoKey(video) === selectedKey && 'bg-muted/70',
              )}
            >
              <span className="min-w-0 flex-1 truncate font-mono text-[13px]">{video.name}</span>
              <Badge variant="outline" className="h-[18px] px-1.5 text-[11px] uppercase">
                {video.source}
              </Badge>
              <span className="text-muted-foreground tabular w-14 text-right text-[11px]">
                {formatClock(video.durationSeconds * MS_PER_SECOND)}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </>
  )
}
