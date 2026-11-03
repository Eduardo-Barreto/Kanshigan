import './pipeline.css'
import { slide, Step } from '@nimbus/deck'

type Stage = { name: string; detail: string }

const PREPARE: Stage[] = [
  { name: 'Decodificação', detail: 'vídeo em resolução e taxa nativas' },
  { name: 'Detecção do dohyo', detail: 'limiarização e ajuste de elipse' },
  { name: 'Calibração', detail: 'escala cm/pixel pelo eixo maior da elipse' },
  { name: 'Recorte', detail: 'retângulo fixo com margem em volta da arena' },
]

const EXTRACT: Stage[] = [
  { name: 'Detecção dos robôs', detail: 'YOLOv8s ou YOLO26n, fine-tuning no domínio' },
  { name: 'Rastreamento', detail: 'quatro rastreadores sobre as mesmas detecções' },
  { name: 'Métricas', detail: 'posição e velocidade no referencial da arena' },
  { name: 'Eventos', detail: 'início, contato e ring-out por regra' },
]

type RowProps = { label: string; stages: Stage[]; at: number; start: number }

function Row({ label, stages, at, start }: RowProps) {
  return (
    <Step at={at} className="pipe-row">
      <p className="pipe-label">{label}</p>
      <ol className="rail pipe-stages" data-numbered style={{ '--start': start }}>
        {stages.map((stage) => (
          <li key={stage.name}>
            <b>{stage.name}</b>
            <span>{stage.detail}</span>
          </li>
        ))}
      </ol>
    </Step>
  )
}

function Pipeline() {
  return (
    <div className="page pipe">
      <p className="eyebrow">Metodologia · pesquisa experimental</p>
      <h2 className="headline">Oito estágios, cada um com saída que dá pra inspecionar.</h2>
      <Row label="Preparação do quadro" stages={PREPARE} at={0} start={0} />
      <Row label="Extração de informação" stages={EXTRACT} at={1} start={PREPARE.length} />
      <Step at={1} delay={0.3} className="pipe-offline">
        <span className="pipe-offline-tag">offline</span>
        <p>
          <b>Anotação offline:</b> o SAM 3 gera o treino, e a revisão à mão fica no gold. Fora da
          inferência.
        </p>
      </Step>
    </div>
  )
}

export const pipeline = slide({
  id: 'pipeline',
  title: 'Pipeline',
  steps: 2,
  notes: (
    <>
      <p>
        (45s) A pesquisa é experimental: a gente compara alternativas medindo cada uma contra um
        conjunto de referência revisado à mão, o gold. Quem sustenta os experimentos é essa
        pipeline. A primeira fileira prepara o quadro: decodifica, acha a arena, tira a escala e
        recorta.
      </p>
      <p>
        [avançar] A segunda extrai a informação: detecta os robôs, rastreia, mede e acha os eventos.
        E embaixo, a anotação dos dados de treino, que acontece offline. Separar em estágios deixa
        avaliar cada peça sozinha. Vou passar pelas que mais pesam.
      </p>
    </>
  ),
  Component: Pipeline,
})
