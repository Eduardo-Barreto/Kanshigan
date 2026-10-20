import type { Role } from '@/shared/clip'

/** How the studio presents one annotation directory under `data/annotations`. */
export type DatasetProfile = { role: Role; label: string; description: string }

const KNOWN_DATASETS: Record<string, DatasetProfile> = {
  gold_crop: {
    role: 'gold',
    label: 'Gold BR',
    description: 'Round BR de teste, recortado no dohyo. É o gold que o artigo reporta.',
  },
  gold_jp: {
    role: 'gold',
    label: 'Gold JP',
    description: 'Round JP de teste, câmera fixa.',
  },
  gold_candidates: {
    role: 'gold_candidate',
    label: 'Candidatos a gold',
    description: 'Partidas anotadas pelo SAM 3 que ainda esperam revisão e aprovação.',
  },
  train: {
    role: 'train',
    label: 'Treino',
    description: 'Rótulos do SAM 3 que treinam o detector reportado.',
  },
  candidates_jp: {
    role: 'train',
    label: 'Treino JP revisado',
    description: 'Rounds JP adicionais, revisados quadro a quadro.',
  },
  val: {
    role: 'val',
    label: 'Validação',
    description: 'Rounds usados para parar o treino.',
  },
  gold: {
    role: 'other',
    label: 'Gold BR nativo',
    description: 'Origem do Gold BR, em quadro inteiro, antes do recorte no dohyo.',
  },
}

const UNKNOWN_DATASET: Omit<DatasetProfile, 'label'> = {
  role: 'other',
  description: 'Conjunto de experimento, fora do treino e da avaliação.',
}

const ROLE_ORDER: Role[] = ['gold', 'gold_candidate', 'train', 'val', 'other']

/** Resolves how a dataset directory is presented, falling back to its name for unknown ones. */
export function datasetProfile(dataset: string): DatasetProfile {
  // Own-property lookup: a dataset named `constructor` must not resolve to Object.prototype.
  const known = Object.hasOwn(KNOWN_DATASETS, dataset) ? KNOWN_DATASETS[dataset] : undefined
  return known ?? { ...UNKNOWN_DATASET, label: dataset }
}

/** Frames of a gold dataset were approved by a human before the studio existed. */
export function defaultReview(role: Role) {
  return role === 'gold' ? 'approved' : 'pending'
}

/** Sort key that lists gold first, then candidates, training data and experiments. */
export function roleRank(role: Role): number {
  return ROLE_ORDER.indexOf(role)
}
