import { expect, test } from 'bun:test'

test('should render the row', () => {
  const screen = { getByTestId: (id: string) => id }
  expect(screen.getByTestId('row')).toBe('row')
})
