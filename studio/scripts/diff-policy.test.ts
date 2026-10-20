import { describe, expect, test } from 'bun:test'

import { deadRelativeLinks, missingTestCoverage, suppressionCount } from './diff-policy.ts'

describe('missing test coverage', () => {
  test('reports a source change with no test change', () => {
    expect(missingTestCoverage(['src/features/members.tsx'])).toEqual(['src/features/members.tsx'])
  })

  test('accepts a source change alongside a unit test', () => {
    expect(
      missingTestCoverage(['src/features/members.tsx', 'src/features/members.test.tsx']),
    ).toEqual([])
  })

  test('accepts a source change alongside an e2e spec', () => {
    expect(missingTestCoverage(['src/routes/dashboard.tsx', 'e2e/dashboard.spec.ts'])).toEqual([])
  })

  test('accepts a component change alongside a tsx spec', () => {
    expect(
      missingTestCoverage(['src/features/members.tsx', 'src/features/members.spec.tsx']),
    ).toEqual([])
  })

  test('ignores the generated route tree', () => {
    expect(missingTestCoverage(['src/routeTree.gen.ts'])).toEqual([])
  })

  test('ignores a diff that touches no source at all', () => {
    expect(missingTestCoverage(['README.md', '.github/workflows/ci.yml'])).toEqual([])
  })
})

const exists = (path: string) => path === 'docs/real.md'

describe('dead relative links', () => {
  test('reports a bare relative target, the form this repo actually writes', () => {
    const content = '[plan](reference/plan.md)'

    expect(deadRelativeLinks([{ path: 'docs/a.md', content }], exists)).toEqual([
      'docs/a.md -> reference/plan.md',
    ])
  })

  test('accepts a bare relative target that resolves', () => {
    const content = '[real](real.md)'

    expect(deadRelativeLinks([{ path: 'docs/a.md', content }], exists)).toEqual([])
  })

  test('leaves a bare anchor alone', () => {
    expect(deadRelativeLinks([{ path: 'docs/a.md', content: '[s](#section)' }], exists)).toEqual([])
  })

  test('leaves a root-absolute path alone', () => {
    expect(deadRelativeLinks([{ path: 'docs/a.md', content: '[r](/gone.md)' }], exists)).toEqual([])
  })

  test('reports a link pointing at a file that does not exist', () => {
    const dead = deadRelativeLinks([{ path: 'docs/a.md', content: '[gone](./missing.md)' }], exists)

    expect(dead).toEqual(['docs/a.md -> ./missing.md'])
  })

  test('accepts a link that resolves', () => {
    expect(deadRelativeLinks([{ path: 'docs/a.md', content: '[ok](./real.md)' }], exists)).toEqual(
      [],
    )
  })

  test('leaves absolute URLs alone', () => {
    const content = '[docs](https://oxc.rs/docs)'

    expect(deadRelativeLinks([{ path: 'docs/a.md', content }], exists)).toEqual([])
  })

  test('resolves a link that walks up a directory', () => {
    const content = '[up](../docs/real.md)'

    expect(deadRelativeLinks([{ path: 'src/a.md', content }], exists)).toEqual([])
  })
})

describe('suppression count', () => {
  test('counts a rule turned off', () => {
    expect(suppressionCount({ rules: { 'eslint/no-console': 'off' } })).toBe(1)
  })

  test('counts every ignored path', () => {
    expect(suppressionCount({ ignorePatterns: ['a/**', 'b/**'] })).toBe(2)
  })

  test('counts rules turned off inside an override', () => {
    const config = { overrides: [{ files: ['x'], rules: { a: 'off', b: 'off', c: 'error' } }] }

    expect(suppressionCount(config)).toBe(2)
  })

  test('counts nothing for a config that suppresses nothing', () => {
    expect(suppressionCount({ rules: { a: 'error', b: ['warn', { max: 3 }] } })).toBe(0)
  })
})
