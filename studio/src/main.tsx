import { QueryClientProvider } from '@tanstack/react-query'
import { RouterProvider, createRouter } from '@tanstack/react-router'
import { ThemeProvider } from 'next-themes'
import { StrictMode, lazy } from 'react'
import { createRoot } from 'react-dom/client'

import { NotFound, RouteError } from '@/components/route-error'
import { Toaster } from '@/components/ui/sonner'
import { queryClient } from '@/lib/query-client'

import { routeTree } from './routeTree.gen'

import './index.css'

const router = createRouter({
  routeTree,
  basepath: import.meta.env.BASE_URL,
  defaultPreload: 'intent',
  defaultErrorComponent: RouteError,
  defaultNotFoundComponent: NotFound,
  scrollRestoration: true,
})

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}

const QueryDevtools = import.meta.env.PROD
  ? () => null
  : lazy(async () => {
      const devtools = await import('@tanstack/react-query-devtools')
      return { default: devtools.ReactQueryDevtools }
    })

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem disableTransitionOnChange>
      <QueryClientProvider client={queryClient}>
        <RouterProvider router={router} />
        <Toaster />
        <QueryDevtools buttonPosition="bottom-left" />
      </QueryClientProvider>
    </ThemeProvider>
  </StrictMode>,
)
