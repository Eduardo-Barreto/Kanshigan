import { describe, expect, test } from 'bun:test'

import type { ColumnDef } from '@tanstack/react-table'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { DataTable } from './data-table'

type Row = { name: string }

const columns: ColumnDef<Row>[] = [{ accessorKey: 'name', header: 'Name' }]
// A header that can actually be clicked to sort, which is what puts aria-sort on one header
// and leaves it off the rest.
const sortableColumns: ColumnDef<Row>[] = [
  {
    accessorKey: 'name',
    header: ({ column }) => (
      <button type="button" onClick={() => column.toggleSorting(false)}>
        Name
      </button>
    ),
  },
  { accessorKey: 'note', header: 'Note' },
]
const rows: Row[] = Array.from({ length: 15 }, (_, index) => ({
  name: `Member ${index + 1}`,
}))

describe('DataTable', () => {
  test('shows the first page of rows', () => {
    render(<DataTable columns={columns} data={rows} label="Members" />)

    expect(screen.getByRole('cell', { name: 'Member 1' })).toBeInTheDocument()
    expect(screen.queryByRole('cell', { name: 'Member 11' })).not.toBeInTheDocument()
  })

  test('reveals later rows when the reader pages forward', async () => {
    const user = userEvent.setup()
    render(<DataTable columns={columns} data={rows} label="Members" />)

    await user.click(screen.getByRole('button', { name: 'Próximo' }))

    expect(screen.getByRole('cell', { name: 'Member 11' })).toBeInTheDocument()
  })

  test('narrows rows to those matching the filter', async () => {
    const user = userEvent.setup()
    render(
      <DataTable
        columns={columns}
        data={rows}
        label="Members"
        filterColumn="name"
        filterPlaceholder="Filter members"
      />,
    )

    await user.type(screen.getByLabelText('Filter members'), 'Member 12')

    expect(screen.getByText('Member 12')).toBeInTheDocument()
    expect(screen.queryByText('Member 1')).not.toBeInTheDocument()
  })

  test('reports an empty state when no rows match', async () => {
    const user = userEvent.setup()
    render(
      <DataTable
        columns={columns}
        data={rows}
        label="Members"
        filterColumn="name"
        filterPlaceholder="Filter members"
      />,
    )

    await user.type(screen.getByLabelText('Filter members'), 'nobody')

    expect(screen.getByText('Nenhum resultado.')).toBeInTheDocument()
  })

  test('gives the table an accessible name', () => {
    render(<DataTable columns={columns} data={rows} label="Members" />)

    expect(screen.getByRole('table', { name: 'Members' })).toBeInTheDocument()
  })

  test('leaves aria-sort off every header while nothing is sorted', () => {
    render(<DataTable columns={columns} data={rows} label="Members" />)

    expect(screen.getByRole('columnheader', { name: 'Name' })).not.toHaveAttribute('aria-sort')
  })

  test('marks only the sorted header once the reader sorts', async () => {
    const user = userEvent.setup()
    render(<DataTable columns={sortableColumns} data={rows} label="Members" />)

    await user.click(screen.getByRole('button', { name: 'Name' }))

    expect(screen.getByRole('columnheader', { name: 'Name' })).toHaveAttribute(
      'aria-sort',
      'ascending',
    )
    expect(screen.getByRole('columnheader', { name: 'Note' })).not.toHaveAttribute('aria-sort')
  })
})
