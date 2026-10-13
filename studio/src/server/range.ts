const RANGE_HEADER = /^bytes=(?<start>\d*)-(?<end>\d*)$/
const HTTP_PARTIAL_CONTENT = 206
const HTTP_RANGE_NOT_SATISFIABLE = 416

/** An inclusive byte range resolved against a file size. */
export type ByteRange = { start: number; end: number }

function suffixRange(suffixLength: number, size: number): ByteRange | 'unsatisfiable' {
  const length = Math.min(size, suffixLength)
  return length === 0 ? 'unsatisfiable' : { start: size - length, end: size - 1 }
}

/**
 * Resolves a `Range` header against a file size. Returns null for a missing or malformed
 * header (serve the whole file) and `'unsatisfiable'` when the range starts past the end.
 */
export function resolveRange(
  header: string | undefined,
  size: number,
): ByteRange | null | 'unsatisfiable' {
  const groups = RANGE_HEADER.exec(header ?? '')?.groups
  if (groups === undefined || (groups.start === '' && groups.end === '')) return null
  if (groups.start === '') return suffixRange(Number(groups.end), size)
  const start = Number(groups.start)
  const end = groups.end === '' ? size - 1 : Math.min(size - 1, Number(groups.end))
  return start >= size || start > end ? 'unsatisfiable' : { start, end }
}

/** Serves a file honouring `Range`, which a `<video>` element needs in order to seek. */
export function serveWithRange(
  path: string,
  rangeHeader: string | undefined,
  contentType: string,
): Response {
  const file = Bun.file(path)
  const range = resolveRange(rangeHeader, file.size)
  const baseHeaders = { 'Content-Type': contentType, 'Accept-Ranges': 'bytes' }
  if (range === 'unsatisfiable') {
    return new Response(null, {
      status: HTTP_RANGE_NOT_SATISFIABLE,
      headers: { ...baseHeaders, 'Content-Range': `bytes */${file.size}` },
    })
  }
  if (range === null) return new Response(file, { headers: baseHeaders })
  return new Response(file.slice(range.start, range.end + 1), {
    status: HTTP_PARTIAL_CONTENT,
    headers: { ...baseHeaders, 'Content-Range': `bytes ${range.start}-${range.end}/${file.size}` },
  })
}
