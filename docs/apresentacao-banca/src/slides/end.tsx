import './end.css'
import { slide, Step } from '@nimbus/deck'

const CONTRIBUTIONS = [
  {
    title: 'Mapa do domínio',
    body: 'seis condições que isolam o par C1 + C2 em vídeo não controlado',
  },
  {
    title: 'Pipeline aberta e medida',
    body: 'dois detectores e quatro rastreadores comparados em acurácia e custo',
  },
  {
    title: 'Anotação com SAM 3',
    body: 'protocolo semiautomático aferido contra revisão humana em 15 rounds JP',
  },
  {
    title: 'Avaliação sem rótulos',
    body: 'ancorada no gold, mostra onde a pipeline falha fora dele',
  },
]

function End() {
  return (
    <div className="page end">
      <p className="eyebrow">Contribuições</p>
      <div className="end-list">
        {CONTRIBUTIONS.map((contribution, index) => (
          <Step key={contribution.title} delay={index * 0.1} className="end-item">
            <span className="end-index">{index + 1}</span>
            <div>
              <p className="end-title">{contribution.title}</p>
              <p className="caption">{contribution.body}</p>
            </div>
          </Step>
        ))}
      </div>
      <div className="end-foot">
        <p className="end-thanks">Obrigado.</p>
        <p className="end-repo">github.com/Eduardo-Barreto/Kanshigan</p>
      </div>
    </div>
  )
}

export const end = slide({
  id: 'fim',
  title: 'Contribuições',
  notes: (
    <p>
      (30s) Fechando com as quatro contribuições. Seis condições que caracterizam o domínio e isolam
      o par C1 + C2. O Kanshigan, uma pipeline aberta, com dois detectores e quatro rastreadores
      comparados em acurácia e custo. Um protocolo de anotação com o SAM 3, aferido contra revisão
      humana. E uma avaliação sem rótulos que mostra onde a pipeline falha. Código, resultados e o
      diário de bordo com cada decisão e cada erro estão no repositório. Obrigado, e fico aberto às
      perguntas.
    </p>
  ),
  Component: End,
})
