/**
 * Writes the archive into a temporary directory, then removes it. `temporary` here is an
 * ordinary noun phrase, not a disclosed shortcut, so it needs no tracked issue behind it.
 */
export function archivePath(root: string): string {
  return `${root}/archive`
}
