import { useMutation, useQueryClient } from '@tanstack/react-query'
import { X } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Meter } from '@/components/window'
import { cancelJob, queryKeys } from '@/lib/api'
import { cn } from '@/lib/utils'
import type { Job, JobStatus } from '@/shared/ingest'

const STATUS_LABEL: Record<JobStatus, string> = {
  queued: 'na fila',
  running: 'rodando',
  succeeded: 'pronto',
  failed: 'falhou',
  cancelled: 'cancelado',
}
const STATUS_TONE: Record<JobStatus, string> = {
  queued: 'text-muted-foreground',
  running: 'text-foreground',
  succeeded: 'text-success',
  failed: 'text-destructive',
  cancelled: 'text-muted-foreground',
}

function JobRow({ job }: { job: Job }) {
  const queryClient = useQueryClient()
  const cancel = useMutation({
    mutationFn: () => cancelJob(job.id),
    onSuccess: (jobs) => queryClient.setQueryData(queryKeys.jobs, jobs),
  })
  const isActive = job.status === 'queued' || job.status === 'running'
  return (
    <li className="flex flex-col gap-1.5 rounded-md px-2 py-2">
      <div className="flex items-center gap-2">
        <span className="min-w-0 flex-1 truncate text-[13px]">{job.title}</span>
        <Badge
          variant="outline"
          className={cn('h-[18px] px-1.5 text-[11px]', STATUS_TONE[job.status])}
        >
          {STATUS_LABEL[job.status]}
        </Badge>
        {isActive ? (
          <Button
            size="icon-xs"
            variant="ghost"
            aria-label={`Cancelar: ${job.title}`}
            onClick={() => cancel.mutate()}
          >
            <X />
          </Button>
        ) : null}
      </div>
      <Meter
        label={`Progresso de ${job.title}`}
        fraction={job.stepCount === 0 ? 0 : job.stepIndex / job.stepCount}
        tone={job.status === 'succeeded' ? 'fill-success' : undefined}
      />
      <p className="text-muted-foreground truncate font-mono text-[11px]">
        {job.error ?? (isActive ? `${job.stepLabel} · ${job.lastLine}` : job.stepLabel)}
      </p>
    </li>
  )
}

/** Background jobs, newest first, each with its progress and the last line it printed. */
export function JobList({ jobs }: { jobs: Job[] }) {
  if (jobs.length === 0) {
    return (
      <p className="text-muted-foreground p-3.5 text-xs">
        Nenhuma tarefa nesta sessão do servidor.
      </p>
    )
  }
  return (
    <ul className="min-h-0 flex-1 overflow-y-auto p-1.5">
      {jobs.map((job) => (
        <JobRow key={job.id} job={job} />
      ))}
    </ul>
  )
}
