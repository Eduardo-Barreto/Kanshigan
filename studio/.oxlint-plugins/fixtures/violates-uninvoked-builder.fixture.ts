import { describe, test } from 'bun:test'

// The builder half of a table form, never invoked, so nothing here is inside a describe block.
// This is the one shape that separates treating `describe.each(table)` as a describe from
// treating it as what it is: an expression that returns one. Without that distinction the
// test below is silently accepted, and both table-form guards can be deleted green.
export const runner = describe.each([test('names a case', () => undefined)])
