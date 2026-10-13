import { useMutation, useQueryClient, useSuspenseQuery } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'
import { FolderInput } from 'lucide-react'
import { Suspense, useState } from 'react'
import { toast } from 'sonner'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import { moveClip, moveOptionsQuery, queryKeys, type ClipLocation } from '@/lib/api'
import type { MoveOptions, Role } from '@/shared/clip'

const ROLE_LABEL: Record<Role, string> = {
  gold: 'gold',
  gold_candidate: 'candidato',
  train: 'treino',
  val: 'validação',
  other: 'experimento',
}
const FROZEN_ROLES = new Set<Role>(['gold', 'train', 'val'])

type MoveOption = MoveOptions['options'][number]

function OptionRow({
  option,
  onMove,
  isMoving,
}: {
  option: MoveOption
  onMove: () => void
  isMoving: boolean
}) {
  const isBlocked = option.blockers.length > 0
  return (
    <li className="border-border flex items-start gap-3 border-b py-3 last:border-b-0">
      <div className="min-w-0 flex-1">
        <p className="flex items-center gap-2 text-[13px] font-medium">
          {option.label}
          <Badge variant="outline" className="h-[18px] px-1.5 text-[11px]">
            {ROLE_LABEL[option.role]}
          </Badge>
        </p>
        {isBlocked ? (
          <ul className="text-muted-foreground mt-1 list-disc pl-4 text-xs">
            {option.blockers.map((blocker) => (
              <li key={blocker}>{blocker}</li>
            ))}
          </ul>
        ) : null}
        {!isBlocked && FROZEN_ROLES.has(option.role) ? (
          <p className="text-warning mt-1 text-xs">
            Muda um conjunto que o artigo reporta. Os números precisam ser recalculados.
          </p>
        ) : null}
      </div>
      <Button size="sm" variant="outline" disabled={isBlocked || isMoving} onClick={onMove}>
        Mover
      </Button>
    </li>
  )
}

function MoveOptionList({ location, onMoved }: { location: ClipLocation; onMoved: () => void }) {
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const { data: options } = useSuspenseQuery(moveOptionsQuery(location))
  const move = useMutation({
    mutationFn: (targetDataset: string) => moveClip(location, targetDataset),
    onSuccess: async (destination) => {
      toast.success(`${destination.clip} agora está em ${destination.dataset}`)
      onMoved()
      queryClient.removeQueries({ queryKey: queryKeys.clip(location) })
      await queryClient.invalidateQueries({ queryKey: queryKeys.library })
      await navigate({ to: '/clips/$dataset/$clip', params: destination })
    },
    onError: (error) => toast.error(error.message),
  })
  return (
    <ul>
      {options.map((option) => (
        <OptionRow
          key={option.dataset}
          option={option}
          isMoving={move.isPending}
          onMove={() => move.mutate(option.dataset)}
        />
      ))}
    </ul>
  )
}

/** Dialog that moves the open clip to another dataset, including promoting a candidate to gold. */
export function MoveDialog({
  location,
  onBeforeOpen,
}: {
  location: ClipLocation
  onBeforeOpen: () => Promise<void>
}) {
  const [isOpen, setIsOpen] = useState(false)
  return (
    <Dialog
      open={isOpen}
      onOpenChange={(open) => {
        if (!open) {
          setIsOpen(false)
          return
        }
        // Blockers are computed on the server, so every pending save has to land first.
        void (async () => {
          await onBeforeOpen()
          setIsOpen(true)
        })()
      }}
    >
      <DialogTrigger asChild>
        <Button size="sm" variant="ghost">
          <FolderInput /> Mover
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Mover {location.clip}</DialogTitle>
          <DialogDescription>
            Leva as imagens, os rótulos e as marcações para outro conjunto. Gold, treino e validação
            só aceitam clips sem quadro pendente.
          </DialogDescription>
        </DialogHeader>
        <Suspense
          fallback={<p className="text-muted-foreground text-sm">Verificando os conjuntos…</p>}
        >
          {isOpen ? <MoveOptionList location={location} onMoved={() => setIsOpen(false)} /> : null}
        </Suspense>
      </DialogContent>
    </Dialog>
  )
}
