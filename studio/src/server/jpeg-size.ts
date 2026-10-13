const JPEG_MARKER_PREFIX = 0xff
const START_OF_IMAGE = 0xd8
const FIRST_FRAME_MARKER = 0xc0
const LAST_FRAME_MARKER = 0xcf
const HUFFMAN_TABLE_MARKER = 0xc4
const JPEG_EXTENSION_MARKER = 0xc8
const ARITHMETIC_CONDITIONING_MARKER = 0xcc
const NON_FRAME_MARKERS = new Set([
  HUFFMAN_TABLE_MARKER,
  JPEG_EXTENSION_MARKER,
  ARITHMETIC_CONDITIONING_MARKER,
])
const HEADER_SCAN_BYTES = 65_536
const MARKER_BYTES = 2
const HEIGHT_OFFSET = 5
const WIDTH_OFFSET = 7
const FRAME_HEADER_BYTES = 9

/** Pixel dimensions of an image. */
export type ImageSize = { width: number; height: number }

function isFrameMarker(marker: number): boolean {
  return (
    marker >= FIRST_FRAME_MARKER && marker <= LAST_FRAME_MARKER && !NON_FRAME_MARKERS.has(marker)
  )
}

/**
 * Reads a JPEG's dimensions from its start-of-frame header, without decoding the image.
 * Returns null when the bytes are not a JPEG or the header is beyond the scanned prefix.
 */
export function parseJpegSize(bytes: Uint8Array): ImageSize | null {
  if (bytes[0] !== JPEG_MARKER_PREFIX || bytes[1] !== START_OF_IMAGE) return null
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  let offset = MARKER_BYTES
  while (offset + FRAME_HEADER_BYTES <= bytes.length) {
    if (bytes[offset] !== JPEG_MARKER_PREFIX) return null
    const marker = bytes[offset + 1] ?? 0
    if (isFrameMarker(marker)) {
      return {
        height: view.getUint16(offset + HEIGHT_OFFSET),
        width: view.getUint16(offset + WIDTH_OFFSET),
      }
    }
    offset += MARKER_BYTES + view.getUint16(offset + MARKER_BYTES)
  }
  return null
}

/** Reads the dimensions of a JPEG file on disk, scanning only its header bytes. */
export async function readJpegSize(path: string): Promise<ImageSize | null> {
  const header = await Bun.file(path).slice(0, HEADER_SCAN_BYTES).bytes()
  return parseJpegSize(header)
}
