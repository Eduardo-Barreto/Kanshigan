import './dataset.css'
import { slide, Step } from '@nimbus/deck'

import samExample from '../assets/figures/jp_sam_960_example.webp'

const SPLITS = [
  { name: 'Treino', rounds: 14, frames: 625, how: 'SAM 3' },
  { name: 'Validação', rounds: 2, frames: 74, how: 'SAM 3' },
  { name: 'Gold', rounds: 2, frames: 116, how: 'SAM 3 + revisão à mão' },
]

function Dataset() {
  return (
    <div className="page dataset">
      <p className="eyebrow">Dataset e anotador</p>
      <h2 className="headline">Não existia dataset da modalidade. A gente construiu um.</h2>
      <div className="dataset-body">
        <div className="dataset-left">
          <p className="caption">
            Duas fontes opostas: <b>IRONCup 2025</b> com câmera de mão oblíqua e{' '}
            <b>torneio japonês</b> com câmera fixa vista de cima. Divisão por round, nunca por
            quadro.
          </p>
          <table className="dataset-table">
            <thead>
              <tr>
                <th scope="col">Conjunto</th>
                <th scope="col">Rounds</th>
                <th scope="col">Quadros</th>
                <th scope="col">Anotação</th>
              </tr>
            </thead>
            <tbody>
              {SPLITS.map((split) => (
                <tr key={split.name}>
                  <th scope="row">{split.name}</th>
                  <td>{split.rounds}</td>
                  <td>{split.frames}</td>
                  <td>{split.how}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="dataset-right">
          <div className="frame dataset-frame">
            <img src={samExample} alt="SAM 3 segmentando os dois robôs na fonte japonesa" />
            <span className="chip">SAM 3 anotando a fonte japonesa</span>
          </div>
          <Step at={1} className="dataset-score">
            <span className="figure">0.96</span>
            <p className="caption">
              F1 do SAM 3 contra a revisão humana no gold BR, 0.91 no JP. Em 14 rounds JP extras cai
              pra 0.75. Roda a ~2 fps com ~7 GB: anotador, nunca na pipeline.
            </p>
          </Step>
        </div>
      </div>
    </div>
  )
}

export const dataset = slide({
  id: 'dataset',
  title: 'Dataset e anotador',
  steps: 2,
  notes: (
    <>
      <p>
        (35s) Pra treinar o detector não existia dataset público. A gente montou um com duas fontes
        escolhidas por serem opostas: o torneio brasileiro, câmera na mão, e o japonês, câmera fixa
        de cima. A divisão é por round, pra quadro vizinho não vazar do treino pro teste. O gold
        nunca entra no treino.
      </p>
      <p>
        [avançar] Quem anota é o SAM 3, da Meta: você passa um conceito em texto, a gente usa "toy",
        e ele devolve as máscaras. É pesado demais pra inferência, mas como anotador serve. Contra a
        revisão humana: 0.96 de F1 no gold BR e 0.91 no JP. Em 14 rounds JP extras ele cai pra 0.75
        e a revisão mudou 71% dos quadros. Os rótulos de treino não foram revisados, e isso é uma
        ameaça que a gente assume.
      </p>
    </>
  ),
  Component: Dataset,
})
