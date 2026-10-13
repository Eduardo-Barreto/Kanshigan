import { appendFile, mkdir } from 'node:fs/promises'
import { join } from 'node:path'

import type { Job, JobKind, RoundWindow } from '@/shared/ingest'

import { log } from './log'
import { Problem } from './problem'

const MAX_REMEMBERED_JOBS = 50
const MAX_ACTIVE_JOBS = 20
const HTTP_TOO_MANY_REQUESTS = 429

/** What a step can do while it runs: report progress lines and observe cancellation. */
type StepContext = {
  emit: (line: string) => void
  signal: AbortSignal
  setRoundWindows: (windows: RoundWindow[]) => void
}

/** One named stage of a job. */
export type JobStep = { label: string; run: (context: StepContext) => Promise<void> }

/** What a caller hands over to start a job. */
export type JobSpec = {
  kind: JobKind
  title: string
  video: Job['video']
  steps: JobStep[]
}

type TrackedJob = { job: Job; spec: JobSpec; abort: AbortController }

function update(entry: TrackedJob, patch: Partial<Job>) {
  entry.job = { ...entry.job, ...patch }
}

function wasCancelled(entry: TrackedJob): boolean {
  return entry.abort.signal.aborted
}

/**
 * Runs jobs one at a time, in arrival order. Annotation holds the only GPU and cutting is
 * CPU-heavy, so nothing is gained by overlapping them. Jobs live in memory: a restart
 * forgets the list but each job's log stays on disk.
 */
export function createJobQueue(logDirectory: string, hiddenPaths: string[] = []) {
  const tracked = new Map<string, TrackedJob>()
  let tail: Promise<void> = Promise.resolve()

  const redact = (line: string) =>
    hiddenPaths.reduce((redacted, path) => redacted.replaceAll(path, '…'), line)

  const record = async (entry: TrackedJob, line: string) => {
    update(entry, { lastLine: redact(line) })
    await mkdir(logDirectory, { recursive: true })
    await appendFile(join(logDirectory, `${entry.job.id}.log`), `${line}\n`)
  }

  const runStep = async (entry: TrackedJob, step: JobStep, stepIndex: number) => {
    update(entry, { stepIndex, stepLabel: step.label })
    await step.run({
      emit: (line) => void record(entry, line),
      signal: entry.abort.signal,
      setRoundWindows: (roundWindows) => update(entry, { roundWindows }),
    })
  }

  const run = async (entry: TrackedJob) => {
    if (wasCancelled(entry)) return
    update(entry, { status: 'running' })
    try {
      for (const [stepIndex, step] of entry.spec.steps.entries()) {
        // oxlint-disable-next-line no-await-in-loop -- steps depend on each other's output
        await runStep(entry, step, stepIndex)
      }
      update(entry, { status: 'succeeded', stepIndex: entry.spec.steps.length })
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      update(entry, {
        status: wasCancelled(entry) ? 'cancelled' : 'failed',
        error: redact(message),
      })
      log.warn('job.failed', { jobId: entry.job.id, kind: entry.job.kind, error: message })
    } finally {
      update(entry, { finishedAt: new Date().toISOString() })
    }
  }

  const forgetOldest = () => {
    const finished = [...tracked.values()].filter((entry) => entry.job.finishedAt !== null)
    finished
      .slice(0, Math.max(0, tracked.size - MAX_REMEMBERED_JOBS))
      .forEach((entry) => tracked.delete(entry.job.id))
  }

  return {
    /** Queues a job and returns it at once, still waiting its turn. */
    enqueue(spec: JobSpec): Job {
      const activeCount = [...tracked.values()].filter(
        (entry) => entry.job.finishedAt === null,
      ).length
      if (activeCount >= MAX_ACTIVE_JOBS) {
        throw new Problem(HTTP_TOO_MANY_REQUESTS, 'fila cheia; espere as tarefas terminarem')
      }
      const entry: TrackedJob = {
        spec,
        abort: new AbortController(),
        job: {
          id: crypto.randomUUID(),
          kind: spec.kind,
          title: spec.title,
          status: 'queued',
          video: spec.video,
          createdAt: new Date().toISOString(),
          finishedAt: null,
          stepIndex: 0,
          stepCount: spec.steps.length,
          stepLabel: spec.steps[0]?.label ?? '',
          lastLine: '',
          error: null,
          roundWindows: null,
        },
      }
      tracked.set(entry.job.id, entry)
      forgetOldest()
      const previous = tail
      tail = (async () => {
        await previous
        await run(entry)
      })()
      return entry.job
    },
    /** Every remembered job, newest first. */
    list(): Job[] {
      return [...tracked.values()].map((entry) => entry.job).toReversed()
    },
    /** Stops a queued or running job. Returns false when there is no such job. */
    cancel(jobId: string): boolean {
      const entry = tracked.get(jobId)
      if (entry === undefined) return false
      entry.abort.abort()
      if (entry.job.status === 'queued') {
        update(entry, { status: 'cancelled', finishedAt: new Date().toISOString() })
      }
      return true
    },
    /** Resolves when every job queued so far has finished. */
    idle: () => tail,
  }
}

/** The job queue's public surface. */
export type JobQueue = ReturnType<typeof createJobQueue>
