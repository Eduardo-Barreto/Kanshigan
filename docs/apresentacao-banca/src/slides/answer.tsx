import './answer.css'
import { Count, slide, Step } from '@nimbus/deck'

function Answer() {
  return (
    <div className="page answer">
      <p className="eyebrow">A resposta</p>
      <h2 className="headline">YOLO ajustado ao domínio, com rastreador só de movimento.</h2>
      <div className="rail answer-cards">
        <Step delay={0.05} className="answer-card">
          <span className="figure">
            <Count from={0} to={0.96} decimals={2} />
          </span>
          <p className="caption">
            de <b>mAP@0.5</b> ou mais nas duas fontes, com qualquer um dos dois detectores
          </p>
        </Step>
        <Step delay={0.15} className="answer-card">
          <span className="figure">
            <Count from={0} to={1} />
          </span>
          <p className="caption">
            <b>troca de identidade</b>, no máximo, em 118 caixas. IDF1 de 0.93 com OC-SORT
          </p>
        </Step>
        <Step delay={0.25} className="answer-card">
          <span className="figure">
            <Count from={0} to={97} /> <small>fps</small>
          </span>
          <p className="caption">
            em <b>1080p</b> numa GPU de notebook, com vídeo de 60. Só na CPU, <b>40 fps</b> com o
            YOLO26n
          </p>
        </Step>
      </div>
      <p className="answer-hw">
        GPU: YOLOv8s ou YOLO26n. Só CPU: YOLO26n. Menos de 300 MiB de memória de GPU.
      </p>
      <p className="source">RTX 4070 Laptop 8 GB · Core Ultra 9 185H</p>
    </div>
  )
}

export const answer = slide({
  id: 'resposta',
  title: 'A resposta',
  notes: (
    <p>
      (30s) Então a resposta: a acurácia não separa as combinações, o custo separa. Um YOLO com
      ajuste fino no recorte da arena, com rastreador só de movimento, OC-SORT ou ByteTrack.
      Detecção acima de 0.96, no máximo uma troca de identidade, e a pipeline inteira, da
      decodificação aos eventos, roda a 97 quadros por segundo em 1080p num notebook, com vídeo de
      60. Com GPU, tanto faz o detector. Só com CPU, o YOLO26n quase dobra a velocidade e chega a
      40. Tudo medido, nada projetado.
    </p>
  ),
  Component: Answer,
})
