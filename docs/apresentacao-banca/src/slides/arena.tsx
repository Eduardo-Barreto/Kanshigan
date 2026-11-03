import './arena.css'
import { slide, Step } from '@nimbus/deck'

import ellipse from '../assets/figures/arena_ellipse_frame.webp'

const ABLATION = [
  { label: 'empata', clips: 17, tone: 'var(--border-strong)' },
  { label: 'ganha', clips: 9, tone: 'var(--accent)' },
  { label: 'perde', clips: 4, tone: 'var(--fail)' },
]
const ABLATION_TOTAL = 30

function Arena() {
  return (
    <div className="page arena">
      <div className="arena-text">
        <p className="eyebrow">Arena e escala</p>
        <h2 className="headline">A arena é fixa por regulamento. Visão clássica resolve.</h2>
        <p className="lede">
          Uma elipse ajustada na borda branca, a cada quadro. Os <b>154 cm</b> do diâmetro oficial
          convertem pixel em centímetro, sem calibrar câmera.
        </p>
        <Step at={1} className="arena-ablation">
          <p className="caption">
            Contra pegar a maior mancha branca, em <b>30 clips</b>:
          </p>
          <div className="arena-bar" role="img" aria-label="17 empates, 9 vitórias e 4 derrotas">
            {ABLATION.map((part) => (
              <span
                key={part.label}
                style={{ flexGrow: part.clips / ABLATION_TOTAL, background: part.tone }}
              >
                {part.clips} {part.label}
              </span>
            ))}
          </div>
          <p className="caption arena-note">
            Decide quando o fundo tem outro brilho: na câmera de mão, variação da escala de 0,002
            contra 0,44.
          </p>
        </Step>
      </div>
      <div className="frame arena-frame">
        <img src={ellipse} alt="Quadro da IRONCup com a elipse da arena detectada em amarelo" />
        <span className="chip">Elipse detectada · câmera de mão</span>
      </div>
    </div>
  )
}

export const arena = slide({
  id: 'arena',
  title: 'Arena e escala',
  steps: 2,
  notes: (
    <>
      <p>
        (35s) Pra achar a arena a gente não treina nada: o dohyo é padronizado, disco escuro com
        borda branca. A gente procura essa borda e ajusta uma elipse, a cada quadro, porque a câmera
        de mão se mexe. E o regulamento fixa o diâmetro em 154 centímetros, então a elipse dá a
        conversão de pixel pra centímetro de graça.
      </p>
      <p>
        [avançar] Depois da pré-banca a gente mediu se essa escolha vale a pena, contra o jeito
        ingênuo de pegar a maior mancha branca. Em 30 clips: empata em 17, ganha em 9, perde em 4.
        Ela decide quando o fundo tem outro candidato brilhante, como na câmera de mão brasileira, e
        empata quando a arena é o único brilho da cena.
      </p>
    </>
  ),
  Component: Arena,
})
