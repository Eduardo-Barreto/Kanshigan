import { expect, it } from 'bun:test'

// A table form with no describe around it. `it.each(table)` and the `it.each(table)(...)` it
// returns both resolve to `it`, so counting the builder as a test reports this one test twice.
// The count assertion is the point; a `toContain` passes either way.
it.each([1, 2])('names the case it runs', (value: number) => {
  expect(value).toBeGreaterThan(0)
})
