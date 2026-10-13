/** Fixture for the naming rules. */
export function themeManager() {
  return null
}

/** Fixture for the vague-name rule. */
export function readRows() {
  const data = [1, 2, 3]
  return data
}

/** Fixture for the static-constant rule. */
export function Panel() {
  const columns = ['name', 'email']
  return <span>{columns.join(', ')}</span>
}

/** Arrow component holding a static constant, which the rule now reaches. */
export const ArrowPanel = () => {
  const headers = ['name', 'email']
  return <span>{headers.join(', ')}</span>
}
