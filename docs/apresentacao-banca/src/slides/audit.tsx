import './results.css'
import './audit.css'
import { slide, Step } from '@nimbus/deck'

import worldsFinal from '../assets/videos/worlds_final.mp4'
import { Clip } from '../clip'

type Source = { name: string; detail: string; both: number; tone: string }

const SOURCES: Source[] = [
  { name: 'Japão', detail: 'câmera fixa', both: 1, tone: 'var(--accent)' },
  { name: 'Brasil, gold', detail: 'IDF1 0.93', both: 0.94, tone: 'var(--text)' },
  { name: 'Sumô RC', detail: 'mediana de 23 rounds', both: 0.73, tone: 'var(--accent)' },
  { name: 'Brasil', detail: 'baixa resolução', both: 0.56, tone: 'var(--robot-b)' },
  { name: 'Mundial', detail: 'trecho da final', both: 0.46, tone: 'var(--robot-b)' },
  { name: 'Mundial', detail: 'outro trecho', both: 0, tone: 'var(--fail)' },
]

function Audit() {
  return (
    <div className="page result audit">
      <p className="eyebrow">Fora do gold</p>
      <h2 className="headline">Sem rótulo, dá pra saber onde a pipeline falha.</h2>
      <div className="result-body">
        <div
          className="chart"
          role="group"
          aria-label="Quadros do round com os dois robôs detectados, por fonte"
        >
          <p className="chart-axis">quadros do round com os dois robôs, de 0 a 100%</p>
          {SOURCES.map((source) => (
            <div key={`${source.name}-${source.detail}`} className="chart-row">
              <span className="chart-name">
                <b>{source.name}</b> {source.detail}
              </span>
              <span className="bar" style={{ '--v': source.both, '--tone': source.tone }}>
                <span />
              </span>
              <span className="chart-value">{Math.round(source.both * 100)}%</span>
            </div>
          ))}
        </div>
        <div className="result-side">
          <p className="caption">
            Durante o round, há sempre dois robôs na arena. O indicador conta robôs, não confere
            caixas. No gold, onde a acurácia é conhecida, ele fica em <b>94%</b>.
          </p>
          <Step at={1} motion="pop" className="frame audit-clip">
            <Clip
              src={worldsFinal}
              label="Final do mundial: no movimento rápido, o detector perde os dois robôs"
              stillAt={2}
            />
            <span className="chip" data-tone="fail">
              Final do mundial · borrão de movimento
            </span>
          </Step>
        </div>
      </div>
    </div>
  )
}

export const audit = slide({
  id: 'auditoria',
  title: 'Auditoria sem rótulos',
  steps: 2,
  notes: (
    <>
      <p>
        (50s) Fora do gold eu não tenho rótulo. Então usei uma regra que o próprio esporte dá:
        durante o round tem exatamente dois robôs na arena. Contei em quantos quadros a pipeline vê
        os dois. Isso conta robôs, não diz se a caixa tá certa, então eu ancoro no gold, onde o IDF1
        é 0.93 e a taxa é 94%. O Japão fica no topo. No Sumô RC, 23 rounds de celular de uma
        categoria fora do treino, a mediana é 73%: dez rounds ficam no nível do gold ou melhor, sete
        ficam abaixo de 50%. É indício de que transfere em parte, não medida de acurácia. Brasil em
        baixa resolução e um trecho da final do mundial caem pra metade. O trecho do Japão e o
        segundo trecho da final vêm de execuções anteriores, cujo vídeo de origem a gente não
        recuperou.
      </p>
      <p>
        [avançar] E na final do mundial, transmissão oficial com arena azul e colisão violenta, o
        modo de falha aparece. No movimento lento o detector acha os dois. No rápido, o borrão
        impede as duas detecções. O limite ali não é a semelhança entre robôs, que carregam
        bandeiras diferentes: é falta de treino com esse borrão.
      </p>
    </>
  ),
  Component: Audit,
})
