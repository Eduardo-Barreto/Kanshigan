import type { ComponentProps, ReactNode } from 'react'

import { cn } from '@/lib/utils'

/** A titled surface on the workspace: the unit every screen is tiled from. */
export function Window({ className, ...props }: ComponentProps<'section'>) {
  return (
    <section
      className={cn(
        'bg-window border-window-border focus-within:border-window-border-focus flex min-h-0 min-w-0 flex-col overflow-hidden rounded-xl border shadow-xs transition-colors',
        className,
      )}
      {...props}
    />
  )
}

type WindowTitleBarProps = {
  icon: ReactNode
  title: string
  subtitle?: string
  children?: ReactNode
  actions?: ReactNode
}

/** Title row of a window: icon, name, a muted qualifier, live chips, and trailing actions. */
export function WindowTitleBar({ icon, title, subtitle, children, actions }: WindowTitleBarProps) {
  return (
    <header className="flex h-[42px] flex-none items-center gap-2 pr-2 pl-3.5 select-none">
      <span className="text-muted-foreground [&_svg]:size-[15px]">{icon}</span>
      <h2 className="text-[13px] font-medium tracking-[-0.005em] whitespace-nowrap">{title}</h2>
      {subtitle ? (
        <span className="text-muted-foreground text-xs whitespace-nowrap">{subtitle}</span>
      ) : null}
      <div className="flex min-w-0 flex-1 items-center gap-1.5 overflow-hidden pl-1.5">
        {children}
      </div>
      {actions ? <div className="flex flex-none items-center gap-1.5">{actions}</div> : null}
    </header>
  )
}

/** A compact label and value pair for a title bar, value set in tabular mono. */
export function Chip({ label, value, tone }: { label: string; value: ReactNode; tone?: string }) {
  return (
    <span className="bg-muted/60 inline-flex h-6 flex-none items-center gap-1.5 rounded-md px-2 text-xs whitespace-nowrap">
      <span className="text-muted-foreground font-mono text-[11px]">{label}</span>
      <span className={cn('tabular font-medium', tone)}>{value}</span>
    </span>
  )
}

/** A keyboard key hint. */
export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="bg-muted text-muted-foreground inline-flex h-5 min-w-5 items-center justify-center rounded px-1.5 font-sans text-xs leading-none font-medium whitespace-nowrap">
      {children}
    </kbd>
  )
}

const PERCENT = 100
const METER_VIEW_BOX = '0 0 100 1'

/** A thin progress bar for a fraction between 0 and 1, announced with its label. */
export function Meter({
  fraction,
  label,
  tone,
}: {
  fraction: number
  label: string
  tone?: string
}) {
  const percent = Math.round(Math.min(1, Math.max(0, fraction)) * PERCENT)
  return (
    <span className="bg-foreground/12 relative block h-1 overflow-hidden rounded-full">
      <meter className="sr-only" aria-label={label} min={0} max={PERCENT} value={percent} />
      <svg
        aria-hidden="true"
        viewBox={METER_VIEW_BOX}
        preserveAspectRatio="none"
        className="absolute inset-0 size-full"
      >
        <rect width={percent} height="1" className={cn('fill-foreground/80', tone)} />
      </svg>
    </span>
  )
}
