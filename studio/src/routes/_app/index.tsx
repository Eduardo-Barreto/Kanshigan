import { createFileRoute } from '@tanstack/react-router'

import { LibraryScreen } from '@/features/library/library-screen'
import { libraryQuery } from '@/lib/api'
import { queryClient } from '@/lib/query-client'

export const Route = createFileRoute('/_app/')({
  loader: () => queryClient.ensureQueryData(libraryQuery),
  component: LibraryScreen,
})
