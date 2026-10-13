const PERCENT = 100
const MS_PER_SECOND = 1000
const SECONDS_PER_MINUTE = 60
const SECONDS_FIELD_WIDTH = 4

/** Formats a count with pt-BR thousands separators. */
export function formatCount(count: number): string {
  return count.toLocaleString('pt-BR')
}

/** Share of `part` in `whole` as a whole percentage, or a dash when there is nothing to divide. */
export function formatShare(part: number, whole: number): string {
  if (whole === 0) return '–'
  return `${Math.round((part / whole) * PERCENT)}%`
}

/** Formats milliseconds as `mm:ss.d`, the clock format used across the timeline. */
export function formatClock(milliseconds: number): string {
  const totalSeconds = milliseconds / MS_PER_SECOND
  const minutes = Math.floor(totalSeconds / SECONDS_PER_MINUTE)
  const seconds = totalSeconds - minutes * SECONDS_PER_MINUTE
  return `${String(minutes).padStart(2, '0')}:${seconds.toFixed(1).padStart(SECONDS_FIELD_WIDTH, '0')}`
}
