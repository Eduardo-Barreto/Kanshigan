import { describe, expect, test } from 'bun:test'

import { serialize } from './serialize'

function gate() {
  const { promise, resolve } = Promise.withResolvers<void>()
  return { opened: promise, open: resolve }
}

describe('serialize', () => {
  test('runs operations with the same key one at a time in arrival order', async () => {
    const events: string[] = []
    const firstGate = gate()
    const first = serialize('clip', async () => {
      events.push('first:start')
      await firstGate.opened
      events.push('first:end')
    })
    const second = serialize('clip', async () => {
      events.push('second:start')
    })
    await Bun.sleep(10)
    expect(events).toEqual(['first:start'])
    firstGate.open()
    await Promise.all([first, second])
    expect(events).toEqual(['first:start', 'first:end', 'second:start'])
  })

  test('returns each operation result', async () => {
    const results = await Promise.all([
      serialize('results', async () => 'a'),
      serialize('results', async () => 'b'),
    ])
    expect(results).toEqual(['a', 'b'])
  })

  test('does not block operations with different keys', async () => {
    const blocker = gate()
    const events: string[] = []
    const slow = serialize('slow', async () => {
      await blocker.opened
      events.push('slow')
    })
    await serialize('fast', async () => {
      events.push('fast')
    })
    expect(events).toEqual(['fast'])
    blocker.open()
    await slow
    expect(events).toEqual(['fast', 'slow'])
  })

  test('runs the next operation after one rejects', async () => {
    const failing = serialize('failing', async () => {
      throw new Error('boom')
    })
    const following = serialize('failing', async () => 'recovered')
    const outcomes = await Promise.allSettled([failing, following])
    expect(outcomes.map((outcome) => outcome.status)).toEqual(['rejected', 'fulfilled'])
    expect(outcomes[1]).toEqual({ status: 'fulfilled', value: 'recovered' })
  })
})
