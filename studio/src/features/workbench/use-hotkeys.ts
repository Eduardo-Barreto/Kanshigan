import { useEffect, useRef } from 'react'

/** Maps a key combination, written like `Shift+KeyW` or `Space`, to what it does. */
export type Hotkeys = Record<string, () => void>

const TEXT_ENTRY_TAGS = new Set(['TEXTAREA', 'SELECT'])
const NON_TEXT_INPUT_TYPES = new Set(['range', 'checkbox', 'radio', 'button', 'submit'])
const CONTROL_SELECTOR = 'button, a[href], input[type="checkbox"], [role="checkbox"]'
const NATIVE_CONTROL_KEYS = new Set(['Tab', 'Enter', 'Space'])
const HELD_KEYS = /^(?:Shift\+)?(?:Arrow(?:Left|Right)|Key[WASD])$/

function combinationOf(event: KeyboardEvent): string {
  return `${event.shiftKey ? 'Shift+' : ''}${event.code}`
}

function isTyping(target: Element): boolean {
  if (target instanceof HTMLInputElement) return !NON_TEXT_INPUT_TYPES.has(target.type)
  return (
    TEXT_ENTRY_TAGS.has(target.tagName) ||
    (target instanceof HTMLElement && target.isContentEditable)
  )
}

/**
 * Whether a key press belongs to the page rather than to the shortcuts: typing in a field,
 * anything inside a dialog, and Tab, Enter or Space on a focused control, which have to keep
 * their native meaning for the workbench to stay usable from the keyboard.
 */
export function belongsToPage(target: EventTarget | null, code: string): boolean {
  if (!(target instanceof Element)) return false
  if (isTyping(target) || target.closest('[role="dialog"]') !== null) return true
  return NATIVE_CONTROL_KEYS.has(code) && target.closest(CONTROL_SELECTOR) !== null
}

/**
 * Binds window-level keyboard shortcuts while the component is mounted. Keys are matched by
 * physical position (`event.code`), so WASD stays WASD on any layout. Only navigation and
 * nudging repeat while held: a held toggle would flip back and forth.
 */
export function useHotkeys(hotkeys: Hotkeys) {
  const latest = useRef(hotkeys)
  latest.current = hotkeys

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.ctrlKey || event.metaKey || event.altKey) return
      if (belongsToPage(event.target, event.code)) return
      const combination = combinationOf(event)
      const action = latest.current[combination]
      if (action === undefined) return
      event.preventDefault()
      if (event.repeat && !HELD_KEYS.test(combination)) return
      action()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])
}
