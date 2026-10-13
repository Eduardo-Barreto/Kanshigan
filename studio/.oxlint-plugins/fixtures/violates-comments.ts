// Esse comentário está em português, então a regra de idioma precisa reclamar dele.
// TODO: finish this without linking an issue
export function undocumented(value: string) {
  return value.trim()
}

function undocumentedIndirect(value: string) {
  return value.toUpperCase()
}

export { undocumentedIndirect }

const undocumentedArrow = (value: string) => value.padStart(2)

export { undocumentedArrow }
