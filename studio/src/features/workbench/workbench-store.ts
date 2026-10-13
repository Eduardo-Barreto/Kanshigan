import { moveBox, resizeBox } from '@/shared/box-editing'
import type {
  Box,
  ClipDetail,
  EventKind,
  Frame,
  ReviewStatus,
  Robot,
  SaveFrame,
  SaveTimeline,
} from '@/shared/clip'
import { assignRobot, isIdentified, propagateIdentity, swapRobots } from '@/shared/identity'

const NUDGE_STEP = 0.004
const ANNOTATION_FPS = 10
const MS_PER_SECOND = 1000
const ROBOTS_PER_ROUND = 2
const PRELOAD_AHEAD = 8

type FrameSave = SaveFrame & { index: number }

/** What the store needs from the outside world: persistence, image warm-up and change notices. */
export type WorkbenchPorts = {
  saveFrames: (frames: FrameSave[]) => Promise<unknown>
  saveTimeline: (timeline: SaveTimeline) => Promise<unknown>
  preloadFrame: (index: number) => void
  onDetailChange: (detail: ClipDetail) => void
}

/** Playback and assistance switches the annotator toggles while working. */
export type WorkbenchOptions = { autoApprove: boolean; propagate: boolean }

/** Everything the workbench renders from. Replaced, never mutated, on each change. */
export type WorkbenchSnapshot = {
  detail: ClipDetail
  position: number
  selected: number | null
  draft: Box[] | null
  /** True while the draft is an identity proposal nobody has confirmed or edited. */
  draftIsProposal: boolean
  clipboard: Box[] | null
  playing: boolean
  speed: number
  options: WorkbenchOptions
  openRoundStart: number | null
  pendingSaves: number
  failedSaves: number
  saveError: string | null
}

/** The frame under the cursor. A clip always has at least one slot, so this never misses. */
export function currentFrame(snapshot: WorkbenchSnapshot): Frame {
  const frame = snapshot.detail.frames[snapshot.position]
  if (frame === undefined) throw new Error('workbench cursor is outside the clip')
  return frame
}

/** Boxes on screen: the unsaved draft when there is one, else what the clip holds. */
export function currentBoxes(snapshot: WorkbenchSnapshot): Box[] {
  return snapshot.draft ?? currentFrame(snapshot).boxes
}

/** Milliseconds from the start of the clip to a frame slot. */
export function timeOfSlot(detail: ClipDetail, slotIndex: number): number {
  if (detail.fps === 0) return (slotIndex / ANNOTATION_FPS) * MS_PER_SECOND
  return ((slotIndex * detail.stride) / detail.fps) * MS_PER_SECOND
}

function lastBoxIndex(boxes: Box[]): number | null {
  return boxes.length > 0 ? boxes.length - 1 : null
}

function sameBoxes(first: Box[], second: Box[]): boolean {
  return JSON.stringify(first) === JSON.stringify(second)
}

/**
 * State and behaviour of one clip's annotation session. Edits to the frame under the
 * cursor stay in a draft and are committed when the cursor leaves, a review is given, or
 * playback advances. Commits apply to the local clip at once and are saved in the
 * background, in order.
 */
export class WorkbenchStore {
  #snapshot: WorkbenchSnapshot
  #listeners = new Set<() => void>()
  #saveQueue: Promise<unknown> = Promise.resolve()
  #playTimer: ReturnType<typeof setInterval> | null = null
  #unsaved: (() => Promise<unknown>)[] = []
  readonly #ports: WorkbenchPorts

  constructor(detail: ClipDetail, ports: WorkbenchPorts) {
    this.#ports = ports
    this.#snapshot = {
      detail,
      position: 0,
      selected: null,
      draft: null,
      draftIsProposal: false,
      clipboard: null,
      playing: false,
      speed: 1,
      options: { autoApprove: false, propagate: true },
      openRoundStart: null,
      pendingSaves: 0,
      failedSaves: 0,
      saveError: null,
    }
    this.#enter(this.#firstLabeledPosition())
  }

  subscribe = (listener: () => void) => {
    this.#listeners.add(listener)
    return () => this.#listeners.delete(listener)
  }

  getSnapshot = () => this.#snapshot

  #set(patch: Partial<WorkbenchSnapshot>) {
    this.#snapshot = { ...this.#snapshot, ...patch }
    for (const listener of this.#listeners) listener()
  }

  #firstLabeledPosition(): number {
    const position = this.#snapshot.detail.frames.findIndex((frame) => frame.hasImage)
    return Math.max(0, position)
  }

  async #drain(): Promise<string | null> {
    while (this.#unsaved.length > 0) {
      const [oldest] = this.#unsaved
      if (oldest === undefined) break
      try {
        // oxlint-disable-next-line no-await-in-loop -- saves must land in the order they were made
        await oldest()
        this.#unsaved.shift()
      } catch (error) {
        return error instanceof Error ? error.message : String(error)
      }
    }
    return null
  }

  /**
   * Queues a save behind every earlier one. A save that fails stays at the head of the
   * queue and is sent again before anything newer, so a later success can never report
   * "saved" over data that did not reach the server.
   */
  #enqueue(save: () => Promise<unknown>) {
    this.#unsaved.push(save)
    this.#set({ pendingSaves: this.#snapshot.pendingSaves + 1 })
    const previous = this.#saveQueue
    this.#saveQueue = (async () => {
      await previous
      const saveError = await this.#drain()
      this.#set({
        pendingSaves: this.#snapshot.pendingSaves - 1,
        failedSaves: this.#unsaved.length,
        saveError,
      })
    })()
  }

  /** Sends again whatever failed to save. */
  retrySaves() {
    this.#enqueue(() => Promise.resolve())
  }

  /** Commits the current frame and resolves once every save has been tried. */
  async settled(): Promise<void> {
    this.flush()
    await this.#saveQueue
  }

  #replaceDetail(detail: ClipDetail) {
    this.#set({ detail })
    this.#ports.onDetailChange(detail)
  }

  #applyFrames(saves: FrameSave[]) {
    const byIndex = new Map(saves.map((save) => [save.index, save]))
    const frames = this.#snapshot.detail.frames.map((frame) => {
      const save = byIndex.get(frame.index)
      if (save === undefined) return frame
      const isKept = save.boxes.length > 0 || save.review !== 'pending'
      return { ...frame, ...save, hasImage: frame.hasImage || isKept }
    })
    this.#replaceDetail({ ...this.#snapshot.detail, frames })
    this.#enqueue(() => this.#ports.saveFrames(saves))
  }

  #commit(review?: ReviewStatus) {
    const frame = currentFrame(this.#snapshot)
    const boxes = currentBoxes(this.#snapshot)
    const nextReview = review ?? frame.review
    if (this.#snapshot.draft === null && nextReview === frame.review) return
    this.#set({ draft: null, draftIsProposal: false })
    this.#applyFrames([{ index: frame.index, boxes, review: nextReview }])
  }

  #proposeIdentity(position: number): Box[] | null {
    const frame = this.#snapshot.detail.frames[position]
    const previous = this.#snapshot.detail.frames[position - 1]
    if (!this.#snapshot.options.propagate || frame === undefined || previous === undefined) {
      return null
    }
    if (frame.boxes.length === 0 || isIdentified(frame.boxes)) return null
    const proposed = propagateIdentity(previous.boxes, frame.boxes)
    return sameBoxes(proposed, frame.boxes) ? null : proposed
  }

  #enter(position: number) {
    const frame = this.#snapshot.detail.frames[position]
    if (frame === undefined) return
    const draft = this.#proposeIdentity(position)
    this.#set({
      position,
      draft,
      draftIsProposal: draft !== null,
      selected: lastBoxIndex(draft ?? frame.boxes),
    })
    this.#snapshot.detail.frames
      .slice(position + 1, position + 1 + PRELOAD_AHEAD)
      .forEach((upcoming) => this.#ports.preloadFrame(upcoming.index))
  }

  #dropProposal() {
    if (this.#snapshot.draftIsProposal) this.#set({ draft: null, draftIsProposal: false })
  }

  /** Moves the cursor to a position in the clip, committing the frame it leaves. */
  goTo(position: number) {
    const last = this.#snapshot.detail.frames.length - 1
    const target = Math.min(last, Math.max(0, position))
    if (target === this.#snapshot.position) return
    // A proposed identity is accepted by stepping to the next frame, the way playback does.
    // Any other move (seeking, going back, jumping) only browses, so it drops the proposal.
    if (target !== this.#snapshot.position + 1) this.#dropProposal()
    this.#commit()
    this.#enter(target)
  }

  step(delta: number) {
    this.pause()
    this.goTo(this.#snapshot.position + delta)
  }

  #jumpToNext(matches: (frame: Frame) => boolean) {
    const { frames } = this.#snapshot.detail
    const next = frames.findIndex(
      (frame, position) => position > this.#snapshot.position && matches(frame),
    )
    if (next !== -1) this.step(next - this.#snapshot.position)
  }

  /** Jumps to the next labeled frame nobody has reviewed yet. */
  nextPending() {
    this.#jumpToNext((frame) => frame.hasImage && frame.review === 'pending')
  }

  /** Jumps to the next labeled frame that does not show exactly two robots. */
  nextAnomaly() {
    this.#jumpToNext(
      (frame) =>
        frame.hasImage && frame.review !== 'rejected' && frame.boxes.length !== ROBOTS_PER_ROUND,
    )
  }

  /** Replaces the draft boxes of the current frame. Any edit pauses playback. */
  edit(boxes: Box[], selected: number | null = this.#snapshot.selected) {
    this.pause()
    this.#set({ draft: boxes, draftIsProposal: false, selected })
  }

  select(selected: number | null) {
    this.pause()
    this.#set({ selected })
  }

  cycleSelection(step: number) {
    const count = currentBoxes(this.#snapshot).length
    if (count === 0) return
    const from = this.#snapshot.selected ?? -step
    this.#set({ selected: (from + step + count) % count })
  }

  #transformSelected(transform: (box: Box) => Box) {
    const { selected } = this.#snapshot
    if (selected === null) return
    const boxes = currentBoxes(this.#snapshot)
    this.edit(boxes.map((box, position) => (position === selected ? transform(box) : box)))
  }

  nudge(directionX: number, directionY: number) {
    this.#transformSelected((box) => moveBox(box, directionX * NUDGE_STEP, directionY * NUDGE_STEP))
  }

  grow(directionWidth: number, directionHeight: number) {
    this.#transformSelected((box) =>
      resizeBox(box, 2 * directionWidth * NUDGE_STEP, 2 * directionHeight * NUDGE_STEP),
    )
  }

  deleteSelected() {
    const { selected } = this.#snapshot
    if (selected === null) return
    const boxes = currentBoxes(this.#snapshot).filter((_box, position) => position !== selected)
    this.edit(boxes, boxes.length > 0 ? Math.min(selected, boxes.length - 1) : null)
  }

  clearBoxes() {
    this.edit([], null)
  }

  copyBoxes() {
    this.#set({ clipboard: currentBoxes(this.#snapshot) })
  }

  pasteBoxes() {
    const { clipboard } = this.#snapshot
    if (clipboard !== null) this.edit(clipboard, lastBoxIndex(clipboard))
  }

  /** Names the selected box. With two boxes on screen the other one takes the opposite robot. */
  assign(robot: Robot) {
    const boxes = currentBoxes(this.#snapshot)
    const target = this.#snapshot.selected ?? lastBoxIndex(boxes)
    if (target !== null) this.edit(assignRobot(boxes, target, robot), target)
  }

  swapIdentities() {
    this.edit(swapRobots(currentBoxes(this.#snapshot)))
  }

  /** Approves every labeled frame still pending, in one save. Returns how many it approved. */
  approvePending(): number {
    this.pause()
    this.#dropProposal()
    this.#commit()
    const saves = this.#snapshot.detail.frames
      .filter((frame) => frame.hasImage && frame.review === 'pending')
      .map((frame) => ({ index: frame.index, boxes: frame.boxes, review: 'approved' as const }))
    if (saves.length > 0) this.#applyFrames(saves)
    return saves.length
  }

  /** Gives the current frame a review status and moves on to the next frame. */
  review(status: ReviewStatus) {
    this.pause()
    const frame = currentFrame(this.#snapshot)
    const isUntouchedGap = !frame.hasImage && currentBoxes(this.#snapshot).length === 0
    if (!isUntouchedGap) this.#commit(status)
    const next = this.#snapshot.position + 1
    if (next < this.#snapshot.detail.frames.length) this.#enter(next)
  }

  /**
   * Carries the current frame's identities forward to the end of the clip, one frame at a
   * time, replacing whatever identity those frames had: this is how a swap gets repaired. A frame without boxes does not break the chain: the last named frame keeps
   * serving as the reference.
   */
  propagateForward() {
    this.pause()
    this.#commit()
    const saves: FrameSave[] = []
    let reference = currentFrame(this.#snapshot).boxes
    for (const frame of this.#snapshot.detail.frames.slice(this.#snapshot.position + 1)) {
      if (frame.boxes.length === 0) continue
      const unnamed = frame.boxes.map((box) => ({ ...box, robot: null }))
      const boxes = propagateIdentity(reference, unnamed)
      if (!sameBoxes(boxes, frame.boxes)) {
        saves.push({ index: frame.index, boxes, review: frame.review })
      }
      reference = boxes
    }
    if (saves.length > 0) this.#applyFrames(saves)
  }

  #tick() {
    const { position, detail, options } = this.#snapshot
    if (position >= detail.frames.length - 1) {
      this.pause()
      return
    }
    const frame = currentFrame(this.#snapshot)
    const approves = options.autoApprove && frame.hasImage && frame.review === 'pending'
    this.#commit(approves ? 'approved' : undefined)
    this.#enter(position + 1)
  }

  #startTimer() {
    const interval = MS_PER_SECOND / (ANNOTATION_FPS * this.#snapshot.speed)
    this.#playTimer = setInterval(() => this.#tick(), interval)
  }

  #stopTimer() {
    if (this.#playTimer !== null) clearInterval(this.#playTimer)
    this.#playTimer = null
  }

  pause() {
    if (!this.#snapshot.playing) return
    this.#stopTimer()
    this.#set({ playing: false })
  }

  togglePlay() {
    if (this.#snapshot.playing) {
      this.pause()
      return
    }
    this.#set({ playing: true })
    this.#startTimer()
  }

  setSpeed(speed: number) {
    this.#set({ speed })
    if (!this.#snapshot.playing) return
    this.#stopTimer()
    this.#startTimer()
  }

  setOption(option: keyof WorkbenchOptions, enabled: boolean) {
    this.#set({ options: { ...this.#snapshot.options, [option]: enabled } })
  }

  #saveTimeline(timeline: SaveTimeline) {
    this.#replaceDetail({ ...this.#snapshot.detail, ...timeline })
    this.#enqueue(() => this.#ports.saveTimeline(timeline))
  }

  /** Marks an event at the current frame, or removes it when the same event is already there. */
  toggleEvent(kind: EventKind) {
    const { detail } = this.#snapshot
    const frameIndex = currentFrame(this.#snapshot).index
    const isSame = (event: ClipDetail['events'][number]) =>
      event.kind === kind && event.frameIndex === frameIndex
    const selectedBox = currentBoxes(this.#snapshot)[this.#snapshot.selected ?? -1]
    const robot = kind === 'ring_out' ? (selectedBox?.robot ?? null) : null
    const events = detail.events.some(isSame)
      ? detail.events.filter((event) => !isSame(event))
      : [...detail.events, { kind, frameIndex, robot }]
    this.#saveTimeline({ events, rounds: detail.rounds })
  }

  /** First call opens a round at the current frame; the second closes it and saves the round. */
  markRound() {
    const frameIndex = currentFrame(this.#snapshot).index
    const { openRoundStart, detail } = this.#snapshot
    if (openRoundStart === null) {
      this.#set({ openRoundStart: frameIndex })
      return
    }
    const round = {
      start: Math.min(openRoundStart, frameIndex),
      end: Math.max(openRoundStart, frameIndex),
    }
    this.#set({ openRoundStart: null })
    const isDuplicate = detail.rounds.some(
      (existing) => existing.start === round.start && existing.end === round.end,
    )
    if (!isDuplicate)
      this.#saveTimeline({ events: detail.events, rounds: [...detail.rounds, round] })
  }

  cancelOpenRound() {
    this.#set({ openRoundStart: null })
  }

  removeEvent(position: number) {
    const { detail } = this.#snapshot
    this.#saveTimeline({
      events: detail.events.filter((_event, index) => index !== position),
      rounds: detail.rounds,
    })
  }

  removeRound(position: number) {
    const { detail } = this.#snapshot
    this.#saveTimeline({
      events: detail.events,
      rounds: detail.rounds.filter((_round, index) => index !== position),
    })
  }

  /** Commits the frame under the cursor without moving. Called before leaving the workbench. */
  flush() {
    this.pause()
    this.#dropProposal()
    this.#commit()
  }
}
