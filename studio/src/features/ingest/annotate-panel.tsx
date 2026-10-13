import { zodResolver } from '@hookform/resolvers/zod'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Sparkles } from 'lucide-react'
import { useForm } from 'react-hook-form'
import { toast } from 'sonner'
import { z } from 'zod'

import { Button } from '@/components/ui/button'
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form'
import { Input } from '@/components/ui/input'
import { queryKeys, startAnnotation } from '@/lib/api'
import { safeNameSchema } from '@/shared/clip'
import type { RawVideo, RoundWindow } from '@/shared/ingest'

const JP_SCORE_THRESHOLD = '0.15'
const DEFAULT_DATASET = 'gold_candidates'

const annotateFormSchema = z.object({
  dataset: safeNameSchema,
  clipPrefix: safeNameSchema,
  scoreThreshold: z
    .string()
    .refine((value) => value === '' || (Number(value) > 0 && Number(value) < 1), {
      message: 'Use um número entre 0 e 1, ou deixe vazio',
    }),
})
type AnnotateForm = z.infer<typeof annotateFormSchema>

function defaultsFor(video: RawVideo): AnnotateForm {
  return {
    dataset: DEFAULT_DATASET,
    clipPrefix: video.name.replace(/^yt_/, '').replace(/^[^A-Za-z0-9]+/, ''),
    scoreThreshold: video.source === 'jp' ? JP_SCORE_THRESHOLD : '',
  }
}

type AnnotatePanelProps = {
  video: RawVideo
  windows: RoundWindow[]
  candidateDatasets: string[]
}

/** Right window of the ingest screen: where the clips go, how they are named, and the SAM 3 run. */
export function AnnotatePanel({ video, windows, candidateDatasets }: AnnotatePanelProps) {
  const queryClient = useQueryClient()
  const form = useForm<AnnotateForm>({
    resolver: zodResolver(annotateFormSchema),
    defaultValues: defaultsFor(video),
  })
  const annotate = useMutation({
    mutationFn: (values: AnnotateForm) =>
      startAnnotation({
        source: video.source,
        video: video.name,
        dataset: values.dataset,
        clipPrefix: values.clipPrefix,
        windows,
        scoreThreshold: values.scoreThreshold === '' ? null : Number(values.scoreThreshold),
      }),
    onSuccess: () => {
      toast.success('Na fila. O SAM 3 leva cerca de um minuto por round.')
      return queryClient.invalidateQueries({ queryKey: queryKeys.jobs })
    },
    onError: (error) => toast.error(error.message),
  })

  return (
    <Form {...form}>
      <form
        className="border-border flex flex-none flex-col gap-3 border-b px-3.5 pb-3.5"
        onSubmit={(event) => void form.handleSubmit((values) => annotate.mutate(values))(event)}
      >
        <FormField
          control={form.control}
          name="dataset"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Conjunto de destino</FormLabel>
              <FormControl>
                <Input list="candidate-datasets" className="font-mono" {...field} />
              </FormControl>
              <datalist id="candidate-datasets">
                {candidateDatasets.map((dataset) => (
                  <option key={dataset} value={dataset}>
                    {dataset}
                  </option>
                ))}
              </datalist>
              <FormDescription>
                Gold, treino e validação são congelados e não aceitam clips novos.
              </FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="clipPrefix"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Prefixo dos clips</FormLabel>
              <FormControl>
                <Input className="font-mono" {...field} />
              </FormControl>
              <FormDescription>
                Os clips saem como <span className="font-mono">{field.value || 'prefixo'}_r01</span>
                , <span className="font-mono">_r02</span> e assim por diante.
              </FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="scoreThreshold"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Limiar de detecção do SAM 3</FormLabel>
              <FormControl>
                <Input
                  inputMode="decimal"
                  placeholder="padrão do SAM"
                  className="tabular"
                  {...field}
                />
              </FormControl>
              <FormDescription>
                Vazio usa o padrão. Robôs pequenos e pretos, como nos vídeos JP, precisam de 0,15.
              </FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />
        <Button type="submit" disabled={windows.length === 0 || annotate.isPending}>
          <Sparkles />
          {windows.length === 0
            ? 'Escolha ao menos um round'
            : `Cortar e anotar ${windows.length} ${windows.length === 1 ? 'round' : 'rounds'}`}
        </Button>
      </form>
    </Form>
  )
}
