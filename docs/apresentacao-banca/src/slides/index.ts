import type { SlideDef } from '@nimbus/deck'

import { answer } from './answer'
import { arena } from './arena'
import { audit } from './audit'
import { cover } from './cover'
import { crop } from './crop'
import { dataset } from './dataset'
import { design } from './design'
import { detection } from './detection'
import { end } from './end'
import { gap } from './gap'
import { limits } from './limits'
import { pipeline } from './pipeline'
import { problem } from './problem'
import { question } from './question'
import { tracking } from './tracking'
import { videos } from './videos'

export const SLIDES: SlideDef[] = [
  cover,
  problem,
  gap,
  question,
  pipeline,
  arena,
  crop,
  dataset,
  design,
  detection,
  tracking,
  answer,
  videos,
  audit,
  limits,
  end,
]
