import { useSuspenseQuery } from '@tanstack/react-query'

import { libraryQuery } from '@/lib/api'
import type { Library } from '@/shared/clip'

function totals(library: Library) {
  const clips = library.datasets.flatMap((dataset) => dataset.clips)
  const gold = clips.filter((clip) => clip.role === 'gold')
  return {
    clipCount: clips.length,
    frameCount: clips.reduce((total, clip) => total + clip.frameCount, 0),
    goldFrameCount: gold.reduce((total, clip) => total + clip.frameCount, 0),
  }
}

function StatusItem({ label, value }: { label: string; value: number }) {
  return (
    <span className="inline-flex items-center gap-1.5 px-1.5 whitespace-pre">
      {label}
      <b className="text-foreground tabular font-medium">{value.toLocaleString('pt-BR')}</b>
    </span>
  )
}

/** Footer strip with library-wide counts, always visible under every workspace. */
export function StatusBar() {
  const { data: library } = useSuspenseQuery(libraryQuery)
  const { clipCount, frameCount, goldFrameCount } = totals(library)
  return (
    <footer className="text-muted-foreground flex items-center gap-1.5 px-2 text-xs">
      <StatusItem label="Clips" value={clipCount} />
      <StatusItem label="Quadros anotados" value={frameCount} />
      <StatusItem label="Quadros gold" value={goldFrameCount} />
      <span className="flex-1" />
      <span className="tabular px-1.5">data/annotations</span>
    </footer>
  )
}
