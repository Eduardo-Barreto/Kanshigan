import './design.css'
import { slide, Step } from '@nimbus/deck'

const COLUMNS = [
  {
    head: 'O que comparamos',
    items: [
      'YOLOv8s contra YOLO26n, cinco vezes menor',
      'YOLO sem treino no domínio como linha de base',
      'OC-SORT e ByteTrack (só movimento) contra DeepOCSORT e BoT-SORT (com aparência)',
    ],
  },
  {
    head: 'O que fica fixo',
    items: [
      'Os detectores treinam no mesmo dataset',
      'Os quatro rastreadores recebem exatamente as mesmas detecções',
    ],
  },
  {
    head: 'Contra o quê',
    items: [
      'Gold: rounds fora do treino, revisados à mão',
      'Acurácia e custo medidos no mesmo notebook, uma RTX 4070 Laptop de 8 GB',
    ],
  },
]

function Design() {
  return (
    <div className="page design">
      <p className="eyebrow">Desenho do experimento</p>
      <h2 className="headline">Troca uma peça por vez, com o resto fixo.</h2>
      <div className="rail design-cols">
        {COLUMNS.map((column, index) => (
          <Step key={column.head} delay={0.08 * index} className="design-col">
            <h3>{column.head}</h3>
            <ul>
              {column.items.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </Step>
        ))}
      </div>
    </div>
  )
}

export const design = slide({
  id: 'experimento',
  title: 'Desenho do experimento',
  notes: (
    <p>
      (30s) O experimento troca uma peça por vez. Nos detectores, um YOLO padrão contra um cinco
      vezes menor, e o YOLO sem treino como linha de base. Nos rastreadores, dois que usam só
      movimento contra dois que também usam aparência. O que fica fixo: o mesmo dataset de treino, e
      as mesmas detecções pros quatro rastreadores, então qualquer diferença é do rastreador. E tudo
      contra o gold, com custo medido no mesmo notebook.
    </p>
  ),
  Component: Design,
})
