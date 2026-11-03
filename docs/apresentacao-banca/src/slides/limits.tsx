import './limits.css'
import { slide, Step } from '@nimbus/deck'

type Limit = { problem: string; evidence: string; action: string }

const LIMITS: Limit[] = [
  {
    problem: 'Amostra de teste pequena',
    evidence: 'um round de referência por fonte, um só com identidade, de eventos vistos no treino',
    action: 'Ampliar o gold com identidade, em mais rounds e fontes',
  },
  {
    problem: 'Escala instável no celular',
    evidence: 'variação de 0,18 de mediana no RC, até 0,38, contra 0,001 na câmera fixa',
    action: 'Homografia da arena',
  },
  {
    problem: 'Cinemática e eventos sem validação',
    evidence: 'o início do round dispara no quadro 20 com um detector e no 310 com o outro',
    action: 'Anotar eventos num conjunto de calibração separado',
  },
  {
    problem: 'Borrão de transmissão',
    evidence: 'na final do mundial, o detector perde os dois robôs no movimento rápido',
    action: 'Treinar com vídeo de transmissão',
  },
]

function Limits() {
  return (
    <div className="page limits">
      <p className="eyebrow">Limitações</p>
      <h2 className="headline">O que ainda não está resolvido, e o próximo passo de cada um.</h2>
      <div className="limits-list">
        {LIMITS.map((limit, index) => (
          <Step key={limit.problem} delay={index * 0.1} className="limits-row">
            <div>
              <p className="limits-problem">{limit.problem}</p>
              <p className="caption">{limit.evidence}</p>
            </div>
            <span className="limits-arrow" aria-hidden="true">
              →
            </span>
            <p className="limits-action">{limit.action}</p>
          </Step>
        ))}
      </div>
    </div>
  )
}

export const limits = slide({
  id: 'limitacoes',
  title: 'Limitações',
  notes: (
    <p>
      (50s) Quatro limitações, cada uma com o próximo passo. Primeiro, a amostra: a detecção é
      avaliada em um round por fonte, o rastreamento em um round só, e os dois são de eventos que
      aparecem no treino. Por isso eu falo em equivalência e não em ranking. O passo de maior valor
      é ampliar o gold com identidade, e o Studio do repositório já registra isso. Segundo, a escala
      em centímetros: no celular ela varia 18% de mediana, então velocidade em centímetro não é
      confiável ali sem homografia. Terceiro, a pipeline calcula cinemática e eventos, mas a gente
      não validou: o início do round muda de quadro conforme o detector, então hoje isso serve como
      sugestão pra revisão humana. Quarto, o borrão de transmissão, que se resolve com dado de
      treino.
    </p>
  ),
  Component: Limits,
})
