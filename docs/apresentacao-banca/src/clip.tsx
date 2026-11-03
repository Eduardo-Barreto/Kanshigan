import { holdReady, useStill } from '@nimbus/deck'
import { useEffect, useRef } from 'react'

type ClipProps = {
  src: string
  label: string
  /** Second shown when the slide is still (PDF, captures, presenter preview, reduced motion). */
  stillAt?: number
}

/** A muted looping video that holds the deck's ready signal until it has a frame to show. */
export function Clip({ src, label, stillAt = 1 }: ClipProps) {
  const ref = useRef<HTMLVideoElement>(null)
  const still = useStill()

  useEffect(() => {
    const video = ref.current
    if (!video) return undefined
    const release = holdReady()
    const fallback = setTimeout(release, 5000)
    if (still) {
      video.pause()
      video.addEventListener('seeked', release, { once: true })
      video.currentTime = stillAt
    } else if (video.readyState >= 2) {
      release()
    } else {
      video.addEventListener('loadeddata', release, { once: true })
    }
    return () => {
      clearTimeout(fallback)
      release()
    }
  }, [still, stillAt])

  return (
    <video
      ref={ref}
      src={src}
      aria-label={label}
      muted
      loop
      playsInline
      autoPlay={!still}
      preload="auto"
    />
  )
}
