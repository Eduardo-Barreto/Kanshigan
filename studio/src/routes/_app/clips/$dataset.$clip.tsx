import { createFileRoute } from '@tanstack/react-router'

import { WorkbenchScreen } from '@/features/workbench/workbench-screen'
import { clipQuery } from '@/lib/api'
import { queryClient } from '@/lib/query-client'

export const Route = createFileRoute('/_app/clips/$dataset/$clip')({
  // Always fetched: the workbench builds its editing session from this, so a cached copy
  // from an earlier visit must never be the starting point.
  loader: ({ params }) => queryClient.fetchQuery({ ...clipQuery(params), staleTime: 0 }),
  component: WorkbenchRoute,
})

function WorkbenchRoute() {
  const params = Route.useParams()
  return <WorkbenchScreen key={`${params.dataset}/${params.clip}`} location={params} />
}
