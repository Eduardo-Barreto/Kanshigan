import '@nimbus/deck/styles.css'
import './deck.css'
import { Deck } from '@nimbus/deck'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import { SLIDES } from './slides'

const root = document.getElementById('root')
if (!root) throw new Error('index.html sem #root')

createRoot(root).render(
  <StrictMode>
    <Deck id="kanshigan" title="Kanshigan" slides={SLIDES} />
  </StrictMode>,
)
