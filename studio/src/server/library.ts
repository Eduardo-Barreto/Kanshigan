import { stat } from 'node:fs/promises'
import { join } from 'node:path'

import type { DatasetSummary, Library } from '@/shared/clip'

import { listClipNames, loadClip, toSummary } from './clip-read'
import type { StudioConfig } from './config'
import { datasetProfile, roleRank } from './datasets'
import { listDirectory } from './files'

const NON_DATASET_DIRECTORIES = new Set(['round_marks'])

async function isDataset(config: StudioConfig, directory: string): Promise<boolean> {
  if (NON_DATASET_DIRECTORIES.has(directory) || directory.startsWith('.')) return false
  const entry = await stat(join(config.paths.annotations, directory))
  return entry.isDirectory()
}

async function summarizeDataset(config: StudioConfig, dataset: string): Promise<DatasetSummary> {
  const clipNames = await listClipNames(config, dataset)
  const clips = await Promise.all(
    clipNames.map(async (clip) => toSummary(await loadClip(config, { dataset, clip }))),
  )
  return { dataset, ...datasetProfile(dataset), clips }
}

/** Scans every annotation directory into the library view, gold first. */
export async function readLibrary(config: StudioConfig): Promise<Library> {
  const directories = await listDirectory(config.paths.annotations)
  const flags = await Promise.all(directories.map((directory) => isDataset(config, directory)))
  const datasets = await Promise.all(
    directories
      .filter((_directory, position) => flags[position])
      .map((dataset) => summarizeDataset(config, dataset)),
  )
  return {
    datasets: datasets
      .filter((dataset) => dataset.clips.length > 0)
      .toSorted(
        (first, second) =>
          roleRank(first.role) - roleRank(second.role) || first.label.localeCompare(second.label),
      ),
  }
}
