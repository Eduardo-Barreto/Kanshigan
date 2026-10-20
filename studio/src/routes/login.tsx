import { createFileRoute, redirect } from '@tanstack/react-router'

import { LoginScreen } from '@/features/auth/login-screen'
import { sessionQuery } from '@/lib/api'
import { queryClient } from '@/lib/query-client'

export const Route = createFileRoute('/login')({
  beforeLoad: async () => {
    const session = await queryClient.ensureQueryData(sessionQuery)
    if (session.authenticated) throw redirect({ to: '/' })
  },
  component: LoginScreen,
})
