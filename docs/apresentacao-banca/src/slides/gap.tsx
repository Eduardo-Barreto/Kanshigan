import './gap.css'
import { slide, Step, useSlide } from '@nimbus/deck'

type Coverage = 'full' | 'part' | 'none'

type Work = {
  name: string
  cells: [Coverage, Coverage, Coverage, Coverage, Coverage, Coverage]
  regime: string
  ours?: boolean
}

const CONDITIONS = [
  'movimento não linear',
  'aparência uniforme',
  'eventos de poucos quadros',
  'vídeo heterogêneo',
  'análise pós-partida',
  'sem marcador',
]

const WORKS: Work[] = [
  {
    name: 'ByteTrack',
    cells: ['part', 'part', 'none', 'none', 'none', 'full'],
    regime: 'pedestres (MOT17)',
  },
  {
    name: 'OC-SORT',
    cells: ['full', 'part', 'none', 'none', 'none', 'full'],
    regime: 'pedestres (MOT17)',
  },
  {
    name: 'DanceTrack',
    cells: ['full', 'full', 'none', 'none', 'none', 'full'],
    regime: 'câmera estável, dança em grupo',
  },
  {
    name: 'SportsMOT',
    cells: ['part', 'full', 'none', 'none', 'none', 'full'],
    regime: 'basquete, vôlei e futebol',
  },
  {
    name: 'SoccerNet-Tracking',
    cells: ['part', 'full', 'part', 'none', 'part', 'full'],
    regime: 'transmissão profissional',
  },
  {
    name: 'idtracker.ai',
    cells: ['part', 'full', 'none', 'none', 'full', 'full'],
    regime: 'gravação longa, câmera fixa',
  },
  {
    name: 'Jiu-jitsu scoring',
    cells: ['full', 'part', 'part', 'none', 'full', 'full'],
    regime: 'câmera fixa, pose humana',
  },
  {
    name: 'SSL-Vision',
    cells: ['full', 'none', 'full', 'none', 'none', 'none'],
    regime: 'marcador colorido, câmera fixa',
  },
  {
    name: 'Sumô de Robôs',
    cells: ['full', 'full', 'full', 'full', 'full', 'full'],
    regime: 'vídeo de torneio, sem marcador',
    ours: true,
  },
]

const COVERAGE_LABEL: Record<Coverage, string> = {
  full: 'cobre',
  part: 'parcial',
  none: 'não força',
}

function Gap() {
  const { step } = useSlide()
  return (
    <div className="page gap">
      <p className="eyebrow">Problema e lacuna</p>
      <h2 className="headline gap-title">O domínio em seis condições</h2>
      <table className="gap-matrix" data-focus={step >= 1 ? '' : undefined}>
        <thead>
          <tr>
            <th scope="col" className="gap-work">
              Trabalho
            </th>
            {CONDITIONS.map((condition, index) => (
              <th
                key={condition}
                scope="col"
                className="gap-cond"
                data-pair={index < 2 ? '' : undefined}
              >
                <b>C{index + 1}</b>
                <span>{condition}</span>
              </th>
            ))}
            <th scope="col" className="gap-regime">
              regime de captura
            </th>
          </tr>
        </thead>
        <tbody>
          {WORKS.map((work) => (
            <tr key={work.name} data-ours={work.ours ? '' : undefined}>
              <th scope="row" className="gap-work">
                {work.name}
              </th>
              {work.cells.map((coverage, index) => (
                <td
                  key={CONDITIONS[index]}
                  aria-label={COVERAGE_LABEL[coverage]}
                  data-pair={index < 2 ? '' : undefined}
                >
                  <span className="dot" data-c={coverage} title={COVERAGE_LABEL[coverage]} />
                </td>
              ))}
              <td className="gap-regime">{work.regime}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <Step at={1} delay={0.25} className="gap-band">
        <p>
          Quem cobre o par C1 + C2 tem câmera estável ou gravação longa. Em vídeo de torneio, curto
          e não calibrado, e em hardware comum, <b>segue em aberto.</b>
        </p>
      </Step>
      <ul className="gap-key" aria-label="Legenda">
        <li>
          <span className="dot" data-c="full" /> cobre
        </li>
        <li>
          <span className="dot" data-c="part" /> parcial
        </li>
        <li>
          <span className="dot" data-c="none" /> não força
        </li>
      </ul>
    </div>
  )
}

export const gap = slide({
  id: 'lacuna',
  title: 'Matriz e lacuna',
  steps: 2,
  notes: (
    <>
      <p>
        (75s) Fui ver o que a literatura resolve, e cada trabalho próximo me deu uma condição do
        domínio. Os rastreadores genéricos, ByteTrack e OC-SORT, usam Kalman de velocidade
        constante, e robô de Sumô colide, gira e ricocheteia: C1, movimento não linear. O DanceTrack
        mostra que, quando os alvos são iguais, os melhores rastreadores despencam, o ByteTrack cai
        de 63 pra 48 de HOTA: C2, aparência uniforme. Os benchmarks de esporte trazem C3, o contato
        que decide dura poucos quadros, e C4, nosso vídeo é de câmera de mão e não de transmissão. A
        biologia, com o idtracker.ai, rastreia animais idênticos, mas precisa de gravação longa e
        câmera fixa. O jiu-jitsu mostra que análise depois da luta funciona, C5. E a RoboCup resolve
        com marcador colado no robô, que o Sumô não tem, C6. Nenhum vizinho cobre mais de três.
      </p>
      <p>
        [avançar] O recorte científico está nestas duas: alvos de aparência idêntica sob movimento
        não linear. O DanceTrack marca as duas colunas, mas com câmera estável. A biologia resolve
        C2 com gravação longa. Em vídeo de torneio, curto e não calibrado, o melhor equilíbrio segue
        sem resposta. E a gente acha que o par vale pra outras disputas rápidas entre alvos rígidos
        parecidos, como outras categorias de combate de robôs.
      </p>
    </>
  ),
  Component: Gap,
})
