import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { createJobQueue, type JobSpec, type JobStep } from './jobs'

let logDirectory = ''

beforeEach(async () => {
  logDirectory = await mkdtemp(join(tmpdir(), 'studio-jobs-'))
})

afterEach(async () => {
  await rm(logDirectory, { recursive: true, force: true })
})

const specOf = (title: string, steps: JobStep[]): JobSpec => ({
  kind: 'segment',
  title,
  video: null,
  steps,
})

describe('createJobQueue', () => {
  test('runs jobs one at a time in arrival order', async () => {
    const queue = createJobQueue(logDirectory)
    const events: string[] = []
    const release = Promise.withResolvers<void>()
    queue.enqueue(
      specOf('first', [
        {
          label: 'wait',
          run: async () => {
            events.push('first:start')
            await release.promise
            events.push('first:end')
          },
        },
      ]),
    )
    queue.enqueue(
      specOf('second', [
        {
          label: 'go',
          run: async () => {
            events.push('second:start')
          },
        },
      ]),
    )
    await Bun.sleep(10)
    expect(events).toEqual(['first:start'])
    release.resolve()
    await queue.idle()
    expect(events).toEqual(['first:start', 'first:end', 'second:start'])
    expect(queue.list().map((job) => job.status)).toEqual(['succeeded', 'succeeded'])
  })

  test('runs the steps of a job in order and marks it succeeded', async () => {
    const queue = createJobQueue(logDirectory)
    const ran: string[] = []
    const job = queue.enqueue(
      specOf('steps', [
        { label: 'one', run: async () => void ran.push('one') },
        { label: 'two', run: async () => void ran.push('two') },
      ]),
    )
    expect(job.status).toBe('queued')
    await queue.idle()
    const [finished] = queue.list()
    expect(ran).toEqual(['one', 'two'])
    expect(finished?.status).toBe('succeeded')
    expect(finished?.stepIndex).toBe(2)
    expect(finished?.finishedAt).not.toBeNull()
  })

  test('marks a job failed with the error message and still runs later jobs', async () => {
    const queue = createJobQueue(logDirectory)
    const ran: string[] = []
    queue.enqueue(
      specOf('broken', [
        {
          label: 'explode',
          run: async () => {
            throw new Error('ffmpeg exited with 1')
          },
        },
        { label: 'never', run: async () => void ran.push('never') },
      ]),
    )
    queue.enqueue(specOf('healthy', [{ label: 'ok', run: async () => void ran.push('healthy') }]))
    await queue.idle()
    const jobsByTitle = new Map(queue.list().map((job) => [job.title, job]))
    expect(jobsByTitle.get('broken')?.status).toBe('failed')
    expect(jobsByTitle.get('broken')?.error).toBe('ffmpeg exited with 1')
    expect(jobsByTitle.get('healthy')?.status).toBe('succeeded')
    expect(ran).toEqual(['healthy'])
  })

  test('cancels a queued job without running its steps', async () => {
    const queue = createJobQueue(logDirectory)
    const release = Promise.withResolvers<void>()
    const ran: string[] = []
    queue.enqueue(specOf('blocker', [{ label: 'wait', run: () => release.promise }]))
    const waiting = queue.enqueue(
      specOf('waiting', [{ label: 'skipped', run: async () => void ran.push('waiting') }]),
    )
    expect(queue.cancel(waiting.id)).toBe(true)
    release.resolve()
    await queue.idle()
    const cancelled = queue.list().find((job) => job.id === waiting.id)
    expect(cancelled?.status).toBe('cancelled')
    expect(ran).toEqual([])
  })

  test('reports false when cancelling an unknown job', () => {
    const queue = createJobQueue(logDirectory)
    expect(queue.cancel('missing')).toBe(false)
  })

  test('exposes round windows set by a step on the job', async () => {
    const queue = createJobQueue(logDirectory)
    queue.enqueue(
      specOf('windows', [
        {
          label: 'detect',
          run: async ({ setRoundWindows }) => setRoundWindows([{ start: 1, end: 4 }]),
        },
      ]),
    )
    await queue.idle()
    expect(queue.list()[0]?.roundWindows).toEqual([{ start: 1, end: 4 }])
  })

  test('lists the newest job first', async () => {
    const queue = createJobQueue(logDirectory)
    queue.enqueue(specOf('older', []))
    queue.enqueue(specOf('newer', []))
    await queue.idle()
    expect(queue.list().map((job) => job.title)).toEqual(['newer', 'older'])
  })
})
