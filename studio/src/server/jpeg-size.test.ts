import { describe, expect, test } from 'bun:test'

import { parseJpegSize } from './jpeg-size'

const START_OF_IMAGE = [0xff, 0xd8]
const APP0_SEGMENT = [0xff, 0xe0, 0x00, 0x10, ...Array.from({ length: 14 }, () => 0)]

function startOfFrame(width: number, height: number): number[] {
  return [
    0xff,
    0xc0,
    0x00,
    0x0b,
    0x08,
    height >> 8,
    height & 0xff,
    width >> 8,
    width & 0xff,
    0x01,
    0x01,
    0x11,
    0x00,
  ]
}

function jpegBytes(width: number, height: number): Uint8Array {
  return Uint8Array.from([...START_OF_IMAGE, ...APP0_SEGMENT, ...startOfFrame(width, height)])
}

describe('parseJpegSize', () => {
  test('reads width and height from the start-of-frame segment', () => {
    expect(parseJpegSize(jpegBytes(640, 360))).toEqual({ width: 640, height: 360 })
  })

  test('reads dimensions larger than one byte on each axis', () => {
    expect(parseJpegSize(jpegBytes(1920, 1080))).toEqual({ width: 1920, height: 1080 })
  })

  test('reads a header that has no segment before the frame', () => {
    const bytes = Uint8Array.from([...START_OF_IMAGE, ...startOfFrame(32, 16)])
    expect(parseJpegSize(bytes)).toEqual({ width: 32, height: 16 })
  })

  test('returns null for bytes that are not a JPEG', () => {
    expect(parseJpegSize(Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a]))).toBeNull()
  })

  test('returns null for an empty file', () => {
    expect(parseJpegSize(new Uint8Array())).toBeNull()
  })

  test('returns null when the file ends before any frame header', () => {
    expect(parseJpegSize(Uint8Array.from([...START_OF_IMAGE, ...APP0_SEGMENT]))).toBeNull()
  })

  test('returns null when the frame header is cut short', () => {
    const truncated = jpegBytes(640, 360).slice(0, START_OF_IMAGE.length + APP0_SEGMENT.length + 6)
    expect(parseJpegSize(truncated)).toBeNull()
  })
})
