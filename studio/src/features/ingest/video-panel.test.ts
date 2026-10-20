import { describe, expect, test } from 'bun:test'

import { videoKey, videoNameFromFile } from './video-panel'

describe('videoNameFromFile', () => {
  test('drops the extension and keeps a name that is already safe', () => {
    expect(videoNameFromFile('final_ironcup-2025.mp4')).toBe('final_ironcup-2025')
  })

  test('replaces spaces and symbols that cannot be a directory name', () => {
    expect(videoNameFromFile('Final (IRONCup) 2025.MOV')).toBe('Final_IRONCup_2025')
  })

  test('strips leading characters a path segment cannot start with', () => {
    expect(videoNameFromFile('--_round.mp4')).toBe('round')
  })

  test('falls back to a generic name when nothing usable is left', () => {
    expect(videoNameFromFile('試合.mp4')).toBe('video')
  })
})

describe('videoKey', () => {
  test('tells apart videos of the same name in different footage folders', () => {
    expect(videoKey({ source: 'br', name: 'final' })).not.toBe(
      videoKey({ source: 'jp', name: 'final' }),
    )
  })
})
