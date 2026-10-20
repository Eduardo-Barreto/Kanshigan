import { useQueryClient, useSuspenseQuery } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import {
  ArrowLeft,
  CheckCheck,
  ChevronLeft,
  ChevronRight,
  GanttChart,
  Pause,
  Play,
  SlidersHorizontal,
} from 'lucide-react'
import { useEffect, useState, useSyncExternalStore, type MouseEvent } from 'react'

import { Button } from '@/components/ui/button'
import { Chip, Window, WindowTitleBar } from '@/components/window'
import {
  clipQuery,
  frameImageUrl,
  queryKeys,
  saveFrames,
  saveTimeline,
  type ClipLocation,
} from '@/lib/api'
import { formatClock } from '@/lib/format'
import type { ClipDetail } from '@/shared/clip'

import { FrameStage } from './frame-stage'
import { Inspector } from './inspector'
import { MoveDialog } from './move-dialog'
import { Timeline } from './timeline'
import type { Hotkeys } from './use-hotkeys'
import { useHotkeys } from './use-hotkeys'
import {
  WorkbenchStore,
  currentBoxes,
  currentFrame,
  timeOfSlot,
  type WorkbenchSnapshot,
} from './workbench-store'

const LONG_STEP = 10

function hotkeysFor(store: WorkbenchStore): Hotkeys {
  return {
    Space: () => store.togglePlay(),
    ArrowRight: () => store.step(1),
    ArrowLeft: () => store.step(-1),
    'Shift+ArrowRight': () => store.step(LONG_STEP),
    'Shift+ArrowLeft': () => store.step(-LONG_STEP),
    Tab: () => store.cycleSelection(1),
    'Shift+Tab': () => store.cycleSelection(-1),
    KeyW: () => store.nudge(0, -1),
    KeyA: () => store.nudge(-1, 0),
    KeyS: () => store.nudge(0, 1),
    KeyD: () => store.nudge(1, 0),
    'Shift+KeyW': () => store.grow(0, -1),
    'Shift+KeyA': () => store.grow(-1, 0),
    'Shift+KeyS': () => store.grow(0, 1),
    'Shift+KeyD': () => store.grow(1, 0),
    Delete: () => store.deleteSelected(),
    Backspace: () => store.deleteSelected(),
    Enter: () => store.review('approved'),
    KeyX: () => store.review('rejected'),
    KeyC: () => store.clearBoxes(),
    KeyY: () => store.copyBoxes(),
    KeyV: () => store.pasteBoxes(),
    KeyG: () => store.nextPending(),
    KeyN: () => store.nextAnomaly(),
    Digit1: () => store.assign('A'),
    Digit2: () => store.assign('B'),
    KeyQ: () => store.swapIdentities(),
    KeyI: () => store.propagateForward(),
    KeyH: () => store.toggleEvent('round_start'),
    KeyT: () => store.toggleEvent('first_contact'),
    KeyO: () => store.toggleEvent('ring_out'),
    KeyR: () => store.markRound(),
    Escape: () => store.cancelOpenRound(),
  }
}

function useWorkbenchStore(location: ClipLocation, detail: ClipDetail): WorkbenchStore {
  const queryClient = useQueryClient()
  const [store] = useState(
    () =>
      new WorkbenchStore(detail, {
        saveFrames: (frames) => saveFrames(location, frames),
        saveTimeline: (timeline) => saveTimeline(location, timeline),
        preloadFrame: (index) => {
          new Image().src = frameImageUrl(location, index)
        },
        onDetailChange: (changed) => {
          queryClient.setQueryData(queryKeys.clip(location), changed)
          void queryClient.invalidateQueries({ queryKey: queryKeys.library, refetchType: 'none' })
        },
      }),
  )
  useEffect(() => {
    const flush = () => store.flush()
    window.addEventListener('pagehide', flush)
    return () => {
      window.removeEventListener('pagehide', flush)
      flush()
    }
  }, [store])
  return store
}

/**
 * A button clicked with the mouse keeps focus, and Tab, Enter and Space on a focused control
 * keep their native meaning. Releasing focus hands those keys back to the shortcuts. A
 * keyboard activation reports `detail` 0 and is left alone.
 */
function releaseFocusAfterMouseClick(event: MouseEvent<HTMLDivElement>) {
  if (event.detail === 0 || !(event.target instanceof Element)) return
  const control = event.target.closest('button, a')
  if (control instanceof HTMLElement) control.blur()
}

function saveLabel(snapshot: WorkbenchSnapshot): string {
  if (snapshot.pendingSaves > 0) return 'salvando…'
  return snapshot.draft === null ? 'salvo' : 'editado, salva ao sair do quadro'
}

function SaveState({ snapshot, onRetry }: { snapshot: WorkbenchSnapshot; onRetry: () => void }) {
  if (snapshot.saveError !== null) {
    return (
      <span role="alert" className="text-destructive flex items-center gap-2 text-xs">
        {snapshot.failedSaves}{' '}
        {snapshot.failedSaves === 1 ? 'salvamento falhou' : 'salvamentos falharam'}:{' '}
        {snapshot.saveError}
        <Button size="xs" variant="outline" onClick={onRetry}>
          Tentar de novo
        </Button>
      </span>
    )
  }
  return <output className="text-muted-foreground text-xs">{saveLabel(snapshot)}</output>
}

function Transport({ snapshot, store }: { snapshot: WorkbenchSnapshot; store: WorkbenchStore }) {
  return (
    <>
      <SaveState snapshot={snapshot} onRetry={() => store.retrySaves()} />
      <Button
        size="icon-sm"
        variant="ghost"
        aria-label="Quadro anterior"
        onClick={() => store.step(-1)}
      >
        <ChevronLeft />
      </Button>
      <Button
        size="icon-sm"
        variant="outline"
        aria-label={snapshot.playing ? 'Pausar' : 'Reproduzir'}
        onClick={() => store.togglePlay()}
      >
        {snapshot.playing ? <Pause /> : <Play />}
      </Button>
      <Button
        size="icon-sm"
        variant="ghost"
        aria-label="Próximo quadro"
        onClick={() => store.step(1)}
      >
        <ChevronRight />
      </Button>
    </>
  )
}

/** Opens a clip in the workbench, or says so when the clip has no frame to show at all. */
export function WorkbenchScreen({ location }: { location: ClipLocation }) {
  const { data: loadedDetail } = useSuspenseQuery(clipQuery(location))
  if (loadedDetail.frames.length === 0) {
    return (
      <div className="text-muted-foreground flex h-full flex-col items-center justify-center gap-2">
        <p className="text-foreground text-[15px] font-semibold">{location.clip} não tem quadros</p>
        <p className="text-[13px]">
          Não há imagens no conjunto nem vídeo do clip para decodificar.
        </p>
        <Link to="/" className="text-foreground text-[13px] underline underline-offset-4">
          Voltar para a biblioteca
        </Link>
      </div>
    )
  }
  return <Workbench location={location} loadedDetail={loadedDetail} />
}

type WorkbenchProps = { location: ClipLocation; loadedDetail: ClipDetail }

/** The annotation workbench for one clip: the frame, the clip-wide timeline and the inspector. */
function Workbench({ location, loadedDetail }: WorkbenchProps) {
  const store = useWorkbenchStore(location, loadedDetail)
  const snapshot = useSyncExternalStore(store.subscribe, store.getSnapshot)
  useHotkeys(hotkeysFor(store))

  const { detail, position } = snapshot
  const frame = currentFrame(snapshot)
  const boxes = currentBoxes(snapshot)

  return (
    <div
      role="presentation"
      className="grid h-full grid-cols-[minmax(0,1fr)_340px] grid-rows-[minmax(0,1fr)_132px] gap-3.5"
      onClick={releaseFocusAfterMouseClick}
    >
      <Window aria-label="Quadro">
        <WindowTitleBar
          icon={
            <Link to="/" aria-label="Voltar para a biblioteca" className="hover:text-foreground">
              <ArrowLeft />
            </Link>
          }
          title={detail.clip}
          subtitle={detail.dataset}
          actions={
            <>
              <MoveDialog location={location} onBeforeOpen={() => store.settled()} />
              <Transport snapshot={snapshot} store={store} />
            </>
          }
        >
          <Chip label="quadro" value={`${position + 1}/${detail.frames.length}`} />
          <Chip label="t" value={formatClock(timeOfSlot(detail, frame.index))} />
          <Chip label="caixas" value={boxes.length} />
        </WindowTitleBar>
        <div className="flex min-h-0 flex-1 px-3.5 pb-3.5">
          <FrameStage
            imageUrl={frameImageUrl(location, frame.index)}
            imageLabel={`Quadro ${position + 1} de ${detail.clip}`}
            width={detail.width}
            height={detail.height}
            boxes={boxes}
            selected={snapshot.selected}
            onEdit={(edited, selected) => store.edit(edited, selected)}
            onSelect={(selected) => store.select(selected)}
          />
        </div>
      </Window>
      <Window aria-label="Inspetor" className="row-span-2">
        <WindowTitleBar icon={<SlidersHorizontal />} title="Inspetor" />
        <Inspector snapshot={snapshot} store={store} />
      </Window>
      <Window aria-label="Linha do tempo">
        <WindowTitleBar
          icon={<GanttChart />}
          title="Linha do tempo"
          subtitle="um quadro por coluna"
          actions={
            <Button size="sm" variant="ghost" onClick={() => store.approvePending()}>
              <CheckCheck /> Aprovar pendentes
            </Button>
          }
        >
          <Chip
            label="aprovados"
            value={detail.frames.filter((slot) => slot.review === 'approved').length}
          />
          <Chip label="rounds" value={detail.rounds.length} />
          <Chip label="eventos" value={detail.events.length} />
        </WindowTitleBar>
        <Timeline
          frames={detail.frames}
          events={detail.events}
          rounds={detail.rounds}
          position={position}
          openRoundStart={snapshot.openRoundStart}
          onSeek={(target) => store.step(target - position)}
        />
      </Window>
    </div>
  )
}
