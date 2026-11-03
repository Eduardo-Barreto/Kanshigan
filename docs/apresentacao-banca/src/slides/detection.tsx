import './results.css'
import { slide, Step } from '@nimbus/deck'

type Row = { model: string; source: string; map: number; tone: string; at: number }

const ROWS: Row[] = [
  { model: 'YOLO26n', source: 'gold BR', map: 0.968, tone: 'var(--accent)', at: 0 },
  { model: 'YOLO26n', source: 'gold JP', map: 0.993, tone: 'var(--accent)', at: 0 },
  { model: 'YOLOv8s', source: 'gold BR', map: 0.965, tone: 'var(--text-subtle)', at: 0 },
  { model: 'YOLOv8s', source: 'gold JP', map: 0.976, tone: 'var(--text-subtle)', at: 0 },
  { model: 'YOLOv8s sem ajuste', source: 'gold BR', map: 0.757, tone: 'var(--fail)', at: 1 },
  { model: 'YOLOv8s sem ajuste', source: 'gold JP', map: 0.359, tone: 'var(--fail)', at: 1 },
]

function Detection() {
  return (
    <div className="page result">
      <p className="eyebrow">Resultado · detecção</p>
      <h2 className="headline">O modelo cinco vezes menor empata com o grande.</h2>
      <div className="result-body">
        <div className="chart" role="group" aria-label="mAP@0.5 por detector e fonte">
          <p className="chart-axis">mAP@0.5 no gold, de 0 a 1</p>
          {ROWS.map((row) => (
            <Step
              key={`${row.model}-${row.source}`}
              at={row.at}
              motion="sweep"
              className="chart-row"
            >
              <span className="chart-name">
                <b>{row.model}</b> {row.source}
              </span>
              <span className="bar" style={{ '--v': row.map, '--tone': row.tone }}>
                <span />
              </span>
              <span className="chart-value">{row.map.toFixed(3)}</span>
            </Step>
          ))}
        </div>
        <div className="result-side">
          <p className="figure">
            2,4 M <small>parâmetros</small>
          </p>
          <p className="caption">
            no YOLO26n, contra <b>11,1 M</b> no YOLOv8s. Só trocando a semente, o YOLOv8s varia{' '}
            <b>0.019</b> no JP. A diferença entre os dois é dessa ordem: equivalência, não
            superioridade.
          </p>
          <Step at={1} as="p" className="caption result-warn">
            Sem ajuste fino, aceitando qualquer classe COCO como robô, cai pra <b>0.36</b> na câmera
            de cima. O ajuste no domínio decide, não a arquitetura.
          </Step>
        </div>
      </div>
    </div>
  )
}

export const detection = slide({
  id: 'deteccao',
  title: 'Resultado: detecção',
  steps: 2,
  notes: (
    <>
      <p>
        (40s) Detecção. O mAP vai de 0 a 1 e mede se o detector acha os robôs onde eles estão sem
        inventar robô onde não tem. Os dois modelos passam de 0.96 nas duas fontes, apesar das
        câmeras opostas. E o YOLO26n, com um quinto dos parâmetros, empata com o grande. Treinando o
        YOLOv8s com três sementes, ele varia 0.019 no JP, então a diferença entre arquiteturas é
        dessa ordem. Eu não digo que o pequeno é melhor: digo que basta.
      </p>
      <p>
        [avançar] E a linha de base: o YOLOv8s com pesos COCO, sem ajuste fino, mesmo aceitando
        qualquer classe como robô, fica em 0.76 no BR e 0.36 na vista de cima. O ajuste fino leva os
        dois acima de 0.96. O que move a acurácia é o dado do domínio, não o tamanho do modelo.
      </p>
    </>
  ),
  Component: Detection,
})
