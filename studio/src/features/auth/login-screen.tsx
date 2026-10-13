import { zodResolver } from '@hookform/resolvers/zod'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'
import { useForm } from 'react-hook-form'

import { BrandMark } from '@/components/brand-mark'
import { Button } from '@/components/ui/button'
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form'
import { Input } from '@/components/ui/input'
import { Window } from '@/components/window'
import { login, queryKeys } from '@/lib/api'
import { loginRequestSchema, type LoginRequest } from '@/shared/auth'

const EMPTY_LOGIN: LoginRequest = { password: '' }

/** Password gate shown to a browser that reaches the studio from outside the tailnet. */
export function LoginScreen() {
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const form = useForm<LoginRequest>({
    resolver: zodResolver(loginRequestSchema),
    defaultValues: EMPTY_LOGIN,
  })
  const signIn = useMutation({
    mutationFn: (values: LoginRequest) => login(values.password),
    onSuccess: async (session) => {
      queryClient.setQueryData(queryKeys.session, session)
      await navigate({ to: '/' })
    },
    onError: (error) => form.setError('password', { message: error.message }),
  })

  return (
    <main className="flex h-full items-center justify-center p-4">
      <Window className="w-full max-w-sm p-6">
        <h1 className="flex items-center gap-2.5 text-[15px] font-semibold tracking-[-0.01em]">
          <BrandMark className="size-[26px]" />
          Kanshigan Studio
        </h1>
        <p className="text-muted-foreground mt-1.5 text-[13px]">
          Você está fora da tailnet. Entre com a senha do studio.
        </p>
        <Form {...form}>
          <form
            className="mt-5 flex flex-col gap-3"
            onSubmit={(event) => void form.handleSubmit((values) => signIn.mutate(values))(event)}
          >
            <FormField
              control={form.control}
              name="password"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Senha</FormLabel>
                  <FormControl>
                    <Input type="password" autoComplete="current-password" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <Button type="submit" disabled={signIn.isPending}>
              Entrar
            </Button>
          </form>
        </Form>
      </Window>
    </main>
  )
}
