import { Link } from '@tanstack/react-router'
import { ChevronRight } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Meter } from '@/components/window'
import { frameImageUrl } from '@/lib/api'
import { formatCount, formatShare } from '@/lib/format'
import type { ClipSummary } from '@/shared/clip'

const SOURCE_LABEL: Record<ClipSummary['source'], string> = { br: 'BR', jp: 'JP', unknown: '?' }

function Thumbnail({ clip }: { clip: ClipSummary }) {
  if (clip.thumbnailIndex === null) {
    return <span className="bg-muted h-9 w-[72px] flex-none rounded-md" />
  }
  return (
    <img
      src={frameImageUrl(clip, clip.thumbnailIndex)}
      alt=""
      loading="lazy"
      className="bg-muted h-9 w-[72px] flex-none rounded-md object-cover"
    />
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <span className="flex w-[62px] flex-none flex-col items-end leading-tight">
      <span className="tabular text-[13px] font-medium">{value}</span>
      <span className="text-muted-foreground text-[11px]">{label}</span>
    </span>
  )
}

/** One clip as a library row: thumbnail, name, key counts and review progress. Opens the workbench. */
export function ClipRow({ clip }: { clip: ClipSummary }) {
  const kept = clip.frameCount - clip.rejectedCount
  return (
    <Link
      to="/clips/$dataset/$clip"
      params={clip}
      className="group hover:bg-muted/50 focus-visible:ring-ring/60 flex h-[52px] items-center gap-3 rounded-md px-2 outline-none focus-visible:ring-2"
    >
      <Thumbnail clip={clip} />
      <span className="flex min-w-0 flex-1 flex-col leading-tight">
        <span className="flex items-center gap-2">
          <span className="truncate font-mono text-[13px] font-medium">{clip.clip}</span>
          <Badge variant="outline" className="h-[18px] px-1.5 text-[11px]">
            {SOURCE_LABEL[clip.source]}
          </Badge>
        </span>
        <span className="text-muted-foreground tabular text-[11px]">
          {clip.width}×{clip.height}
          {clip.roundCount > 0 ? ` · ${clip.roundCount} rounds` : ''}
          {clip.eventCount > 0 ? ` · ${clip.eventCount} eventos` : ''}
        </span>
      </span>
      <Stat label="quadros" value={formatCount(clip.frameCount)} />
      <Stat label="2 robôs" value={formatShare(clip.bothRobotsCount, kept)} />
      <Stat label="A/B" value={formatShare(clip.identifiedCount, kept)} />
      <span className="flex w-[112px] flex-none flex-col gap-1">
        <span className="text-muted-foreground tabular text-right text-[11px] whitespace-nowrap">
          {formatCount(clip.approvedCount)}/{formatCount(clip.frameCount)} revisados
        </span>
        <Meter
          label={`Revisão de ${clip.clip}`}
          fraction={clip.frameCount === 0 ? 0 : clip.approvedCount / clip.frameCount}
          tone={clip.approvedCount === clip.frameCount ? 'fill-success' : undefined}
        />
      </span>
      <ChevronRight className="text-muted-foreground size-4 flex-none opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100" />
    </Link>
  )
}
