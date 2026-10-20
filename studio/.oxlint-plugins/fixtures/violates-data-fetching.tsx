import { useQuery } from '@tanstack/react-query'
import { useEffect } from 'react'
import { useForm } from 'react-hook-form'

export function Bad() {
  useEffect(() => {
    void fetch('/api/rows')
  }, [])
  const { isLoading } = useQuery({ queryKey: ['rows'], queryFn: () => Promise.resolve([]) })
  useForm()
  return <div>{isLoading && <span>loading</span>}</div>
}

export async function readPayload(response: Response) {
  return await response.json()
}

export const reader = {
  /** Method reading a payload without validating it. */
  async load(response: Response) {
    return await response.json()
  },
}
