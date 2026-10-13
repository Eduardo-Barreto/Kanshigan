import { mkdir, rename, rm } from 'node:fs/promises'
import { dirname, join } from 'node:path'

import type { FootageSource, Role } from '@/shared/clip'

import { clipPaths, frameStem, type ClipLocation } from './clip-layout'
import { listClipNames, reviewOf, type LoadedClip } from './clip-read'
import type { StudioConfig } from './config'
import { datasetProfile } from './datasets'
import { conflict, unprocessable } from './problem'
import { writeSidecar, type Sidecar } from './sidecar'

const GOLD_DATASET_BY_SOURCE: Partial<Record<FootageSource, string>> = {
  br: 'gold_crop',
  jp: 'gold_jp',
}
const REVIEW_REQUIRED_ROLES = new Set<Role>(['gold', 'train', 'val'])

async function isCroppedToDohyo(clip: LoadedClip): Promise<boolean> {
  if (clip.video === null) return false
  return Bun.file(clip.video.path.replace(/\.mp4$/, '.roi.json')).exists()
}

function pendingFrameCount(clip: LoadedClip): number {
  return [...clip.placements.keys()].filter((index) => reviewOf(clip, index) === 'pending').length
}

/**
 * Reasons a clip cannot join a dataset, in the annotator's terms. Frozen datasets feed
 * training and evaluation directly, so they only take fully reviewed clips; gold also needs
 * the clip in dohyo-crop space and in the gold set of its own footage source.
 */
export async function moveBlockers(clip: LoadedClip, targetDataset: string): Promise<string[]> {
  const targetRole = datasetProfile(targetDataset).role
  const blockers: string[] = []
  const pending = pendingFrameCount(clip)
  if (REVIEW_REQUIRED_ROLES.has(targetRole) && pending > 0) {
    blockers.push(`${pending} quadros ainda estão pendentes de revisão`)
  }
  if (targetRole !== 'gold') return blockers
  if (GOLD_DATASET_BY_SOURCE[clip.source] !== targetDataset) {
    blockers.push(
      `o gold dos vídeos ${clip.source} é ${GOLD_DATASET_BY_SOURCE[clip.source] ?? 'indefinido'}`,
    )
  }
  if (!(await isCroppedToDohyo(clip))) {
    blockers.push('o clip está em quadro inteiro; o gold de avaliação usa o recorte do dohyo')
  }
  return blockers
}

type FileMove = { from: string; to: string }

function frameMoves(clip: LoadedClip, target: ReturnType<typeof clipPaths>): FileMove[] {
  return [...clip.placements].flatMap(([index, placement]) => {
    const stem = frameStem(clip.location.clip, index)
    return [
      {
        from: join(clip.paths.imagesOf(placement), `${stem}.jpg`),
        to: join(target.imagesOf(placement), `${stem}.jpg`),
      },
      {
        from: join(clip.paths.labelsOf(placement), `${stem}.txt`),
        to: join(target.labelsOf(placement), `${stem}.txt`),
      },
    ]
  })
}

async function applyMoves(moves: FileMove[]): Promise<void> {
  const done: FileMove[] = []
  try {
    for (const move of moves) {
      // oxlint-disable-next-line no-await-in-loop -- sequential so a failure knows exactly what to undo
      if (!(await Bun.file(move.from).exists())) continue
      // oxlint-disable-next-line no-await-in-loop -- same
      await mkdir(dirname(move.to), { recursive: true })
      // oxlint-disable-next-line no-await-in-loop -- same
      await rename(move.from, move.to)
      done.push(move)
    }
  } catch (error) {
    await Promise.all(done.map((move) => rename(move.to, move.from)))
    throw error
  }
}

/**
 * The sidecar a clip takes with it. Review statuses that were only implied by the dataset
 * it is leaving (gold frames count as approved) are written out, so the clip means the
 * same thing wherever it lands.
 */
function portableSidecar(clip: LoadedClip): Sidecar {
  const review = { ...clip.sidecar.review }
  for (const index of clip.placements.keys()) review[String(index)] = reviewOf(clip, index)
  return { ...clip.sidecar, review }
}

/**
 * Moves a clip to another dataset: every frame image and label, plus the sidecar, which is
 * written out even when the clip was still relying on imported legacy annotations (those
 * files stay behind in the old dataset). Refuses when the target already has the clip or
 * when a blocker applies.
 */
export async function moveClip(
  config: StudioConfig,
  clip: LoadedClip,
  targetDataset: string,
): Promise<ClipLocation> {
  if (targetDataset === clip.location.dataset) throw unprocessable('o clip já está nesse conjunto')
  if ((await listClipNames(config, targetDataset)).includes(clip.location.clip)) {
    throw conflict(`${targetDataset} já tem um clip chamado ${clip.location.clip}`)
  }
  const blockers = await moveBlockers(clip, targetDataset)
  if (blockers.length > 0) throw unprocessable(blockers.join('; '))

  const destination: ClipLocation = { dataset: targetDataset, clip: clip.location.clip }
  const target = clipPaths(config, destination)
  try {
    await applyMoves(frameMoves(clip, target))
    await writeSidecar(target.sidecar, portableSidecar(clip))
  } catch (error) {
    await rm(target.sidecar, { force: true })
    throw error
  }
  await rm(clip.paths.sidecar, { force: true })
  await Promise.all(
    ['identity.txt', 'events.json', 'rounds.json'].map((suffix) =>
      rm(join(clip.paths.datasetRoot, 'exports', `${clip.location.clip}.${suffix}`), {
        force: true,
      }),
    ),
  )
  return destination
}
