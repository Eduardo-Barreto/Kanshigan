/** Documented at module scope, then shadowed by a nested function of the same name. */
function shadowedExport(value: string) {
  return value.trim()
}

/** Holds a nested binding that reuses the exported name without documenting it. */
export function holder() {
  function shadowedExport(value: string) {
    return value.toUpperCase()
  }
  return shadowedExport
}

export { shadowedExport }
