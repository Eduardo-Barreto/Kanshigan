import { createFileRoute } from '@tanstack/react-router'

import { IngestScreen } from '@/features/ingest/ingest-screen'
import { jobsQuery, libraryQuery, videosQuery } from '@/lib/api'
import { queryClient } from '@/lib/query-client'

export const Route = createFileRoute('/_app/ingest')({
  loader: () =>
    Promise.all([
      queryClient.ensureQueryData(videosQuery),
      queryClient.ensureQueryData(jobsQuery),
      queryClient.ensureQueryData(libraryQuery),
    ]),
  component: IngestScreen,
})
