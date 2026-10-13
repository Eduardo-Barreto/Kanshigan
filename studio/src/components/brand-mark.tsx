/** The studio's mark: a dohyo seen from above, with its two starting lines. */
export function BrandMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 26 26" className={className} aria-hidden="true">
      <rect width="26" height="26" rx="7" className="fill-foreground" />
      <circle cx="13" cy="13" r="7.5" className="stroke-background fill-none" strokeWidth="1.8" />
      <path
        d="M11 10.5v5M15 10.5v5"
        className="stroke-background"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  )
}
