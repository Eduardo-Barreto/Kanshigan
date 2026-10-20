import { zodResolver } from '@hookform/resolvers/zod'
import { useSuspenseQuery } from '@tanstack/react-query'
import { useEffect } from 'react'
import { useForm } from 'react-hook-form'
import { z } from 'zod'

const memberKeys = { all: ['members'] as const }
const memberSchema = z.object({ name: z.string() })
const HEADERS = ['name', 'email']

/** Reads members through a validated boundary, so the payload is typed end to end. */
export async function fetchMembers() {
  const response = await window.fetch('/api/members')
  return memberSchema.array().parse(await response.json())
}

/** Renders the member list with a key factory and a suspense-backed query. */
export function MemberList() {
  const { data: members } = useSuspenseQuery({ queryKey: memberKeys.all, queryFn: fetchMembers })
  useEffect(() => {
    document.title = 'Members'
  }, [])
  return (
    <ul aria-label={HEADERS.join(' ')}>
      {members.map((member) => (
        <li key={member.name}>{member.name}</li>
      ))}
    </ul>
  )
}

/** Form guarded by a Zod resolver. Retry logic tracked in #42 until the API settles. */
export function MemberForm() {
  const form = useForm({ resolver: zodResolver(memberSchema) })
  return <form onSubmit={(event) => void form.handleSubmit(() => {})(event)} />
}
