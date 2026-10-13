import { describe, expect, test } from 'bun:test'

import { belongsToPage } from './use-hotkeys'

function elementFrom(html: string, selector: string): Element {
  document.body.innerHTML = html
  const element = document.querySelector(selector)
  if (element === null) throw new Error(`no ${selector} in the fixture`)
  return element
}

describe('belongsToPage', () => {
  test('leaves typing in a text field alone', () => {
    expect(belongsToPage(elementFrom('<input type="text" />', 'input'), 'KeyW')).toBe(true)
  })

  test('keeps shortcuts live while the timeline slider has focus', () => {
    expect(belongsToPage(elementFrom('<input type="range" />', 'input'), 'KeyW')).toBe(false)
  })

  test('leaves every key alone inside a dialog', () => {
    const button = elementFrom('<div role="dialog"><button>Mover</button></div>', 'button')
    expect(belongsToPage(button, 'KeyX')).toBe(true)
    expect(belongsToPage(button, 'Enter')).toBe(true)
  })

  test('lets Tab, Enter and Space act on a focused button', () => {
    const button = elementFrom('<button>Aprovar</button>', 'button')
    expect(belongsToPage(button, 'Tab')).toBe(true)
    expect(belongsToPage(button, 'Enter')).toBe(true)
    expect(belongsToPage(button, 'Space')).toBe(true)
  })

  test('still runs the other shortcuts from a focused button', () => {
    expect(belongsToPage(elementFrom('<button>Aprovar</button>', 'button'), 'KeyX')).toBe(false)
  })

  test('runs shortcuts when nothing in particular has focus', () => {
    expect(belongsToPage(document.body, 'Tab')).toBe(false)
  })
})
