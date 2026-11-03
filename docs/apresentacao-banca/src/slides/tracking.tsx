import './results.css'
import './tracking.css'
import { slide, Step } from '@nimbus/deck'

type Tracker = { name: string; appearance: boolean; idf1: number; fps: number }

const TRACKERS: Tracker[] = [
  { name: 'OC-SORT', appearance: false, idf1: 0.933, fps: 155 },
  { name: 'ByteTrack', appearance: false, idf1: 0.947, fps: 151 },
  { name: 'DeepOCSORT', appearance: true, idf1: 0.933, fps: 57 },
  { name: 'BoT-SORT', appearance: true, idf1: 0.952, fps: 61 },
]
const FPS_MAX = 155

const tone = (tracker: Tracker) => (tracker.appearance ? 'var(--robot-b)' : 'var(--accent)')

function Tracking() {
  return (
    <div className="page result">
      <p className="eyebrow">Resultado · rastreamento</p>
      <h2 className="headline">Pra manter a identidade, o movimento basta.</h2>
      <div className="track-grid">
        <span />
        <p className="chart-axis">IDF1 com YOLOv8s, de 0 a 1</p>
        <p className="chart-axis track-fps-axis">quadros por segundo, pipeline completa</p>
        {TRACKERS.map((tracker) => (
          <div key={tracker.name} className="track-row">
            <span className="chart-name">
              <b>{tracker.name}</b>
              <span className="pill" data-appearance={tracker.appearance ? '' : undefined}>
                {tracker.appearance ? 'com aparência' : 'só movimento'}
              </span>
            </span>
            <span className="track-cell">
              <span className="bar" style={{ '--v': tracker.idf1, '--tone': tone(tracker) }}>
                <span />
              </span>
              <span className="chart-value">{tracker.idf1.toFixed(3)}</span>
            </span>
            <Step at={1} motion="sweep" className="track-cell">
              <span
                className="bar"
                style={{ '--v': tracker.fps / FPS_MAX, '--tone': tone(tracker) }}
              >
                <span />
              </span>
              <span className="chart-value">{tracker.fps}</span>
            </Step>
          </div>
        ))}
      </div>
      <Step at={1} delay={0.4} as="p" className="caption track-take">
        Mesma identidade, <b>menos da metade da velocidade</b>. Os robôs até são distinguíveis pela
        aparência, mas o movimento já resolve e não sobra erro pra ela corrigir.
      </Step>
    </div>
  )
}

export const tracking = slide({
  id: 'rastreamento',
  title: 'Resultado: rastreamento',
  steps: 2,
  notes: (
    <>
      <p>
        (40s) Rastreamento. O IDF1 mede quanto do tempo cada robô mantém o rótulo certo: o A
        continua A. Os oito pares, dois detectores vezes quatro rastreadores, mantêm a identidade
        com no máximo uma troca em 118 caixas. As diferenças cabem em uma ou duas caixas: o BoT-SORT
        lidera por uma margem que um round só não separa de ruído.
      </p>
      <p>
        [avançar] O custo é outra história: o ReID derruba a pipeline inteira de 128 a 155 pra 54 a
        61 quadros por segundo. Só o passo de rastreamento fica 31 a 35 vezes mais lento. A gente
        esperava que o motivo fosse C2, robôs iguais demais. Medimos e não é: o OSNet separa os dois
        robôs com AUC de 0.94. O movimento sozinho já resolve esse round. Então ele não estressa C2,
        e confronto entre robôs de fato idênticos segue sem medição.
      </p>
    </>
  ),
  Component: Tracking,
})
