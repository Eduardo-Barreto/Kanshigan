import { Link, Outlet, createFileRoute, redirect } from '@tanstack/react-router'
import { Moon, Sun } from 'lucide-react'
import { useTheme } from 'next-themes'
import { Suspense } from 'react'

import { BrandMark } from '@/components/brand-mark'
import { StatusBar } from '@/components/status-bar'
import { Button } from '@/components/ui/button'
import { sessionQuery } from '@/lib/api'
import { queryClient } from '@/lib/query-client'

export const Route = createFileRoute('/_app')({
  beforeLoad: async () => {
    const session = await queryClient.ensureQueryData(sessionQuery)
    if (!session.authenticated) throw redirect({ to: '/login' })
  },
  component: AppLayout,
})

const workspaces = [
  { to: '/', label: 'Biblioteca' },
  { to: '/ingest', label: 'Novo vídeo' },
] as const

const exactMatch = { exact: true }

function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme()
  const isDark = resolvedTheme === 'dark'
  return (
    <Button
      variant="ghost"
      size="icon-sm"
      aria-label={isDark ? 'Usar tema claro' : 'Usar tema escuro'}
      onClick={() => setTheme(isDark ? 'light' : 'dark')}
    >
      {isDark ? <Sun /> : <Moon />}
    </Button>
  )
}

function WorkspaceTabs() {
  return (
    <nav
      aria-label="Áreas"
      className="bg-muted/70 text-muted-foreground absolute top-1/2 left-1/2 inline-flex h-[34px] -translate-x-1/2 -translate-y-1/2 items-center gap-0.5 rounded-[10px] p-[3px]"
    >
      {workspaces.map((workspace) => (
        <Link
          key={workspace.to}
          to={workspace.to}
          activeOptions={exactMatch}
          className="hover:text-foreground focus-visible:ring-ring/50 [&.active]:bg-background [&.active]:text-foreground dark:[&.active]:border-input dark:[&.active]:bg-input/30 inline-flex h-full items-center rounded-[7px] border border-transparent px-[11px] text-[13px] font-medium whitespace-nowrap transition-colors outline-none focus-visible:ring-3 [&.active]:shadow-sm"
        >
          {workspace.label}
        </Link>
      ))}
    </nav>
  )
}

function AppLayout() {
  return (
    <div className="grid h-full grid-rows-[52px_minmax(0,1fr)_32px]">
      <header className="relative z-20 flex items-center gap-2.5 px-3.5">
        <Link to="/" className="flex items-center gap-2.5 text-sm font-semibold tracking-[-0.01em]">
          <BrandMark className="size-[26px]" />
          Kanshigan Studio
        </Link>
        <WorkspaceTabs />
        <div className="ml-auto flex items-center gap-2">
          <ThemeToggle />
        </div>
      </header>
      <main className="min-h-0 px-3.5">
        <Outlet />
      </main>
      <Suspense fallback={<footer />}>
        <StatusBar />
      </Suspense>
    </div>
  )
}
