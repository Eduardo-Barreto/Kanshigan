import './crop.css'
import { slide, Step } from '@nimbus/deck'

import cropped from '../assets/figures/crop_after_boxes.webp'
import raw from '../assets/figures/raw_frame.webp'

function Crop() {
  return (
    <div className="page crop">
      <p className="eyebrow">Recorte e detecção</p>
      <h2 className="headline">Recortar na arena descarta 62% do quadro, plateia junto.</h2>
      <div className="crop-row">
        <figure className="crop-fig">
          <div className="frame crop-raw">
            <img src={raw} alt="Quadro inteiro, com plateia ao fundo" />
          </div>
          <figcaption>
            <span>quadro inteiro</span>
            <b>precisão ~0.71</b>
          </figcaption>
        </figure>
        <span className="crop-arrow" aria-hidden="true" />
        <Step at={1} motion="pop" as="figure" className="crop-fig">
          <div className="frame crop-after">
            <img src={cropped} alt="Recorte na elipse da arena, com os robôs A e B detectados" />
          </div>
          <figcaption>
            <span>recorte no dohyo</span>
            <b className="crop-good">precisão 0.99 · revocação 0.96</b>
          </figcaption>
        </Step>
      </div>
    </div>
  )
}

export const crop = slide({
  id: 'recorte',
  title: 'Recorte e detecção',
  steps: 2,
  notes: (
    <>
      <p>
        (35s) Com a arena na mão, veio o passo que mais mudou o projeto. Reduzido pros 640 pixels do
        detector, o quadro inteiro deixa o robô com poucos pixels, que o borrão apaga, e a plateia e
        as mãos dos operadores viram falso positivo.
      </p>
      <p>
        [avançar] Recortando um retângulo em volta da arena, 62% do quadro sai e o robô fica cerca
        de 1,2 vez maior na entrada. No gold brasileiro, o YOLOv8s no recorte tem precisão 0.99 e
        revocação 0.96. O 0.71 do quadro inteiro foi o que a gente viu no desenvolvimento: os pesos
        daquele detector não ficaram salvos, então é observação, não experimento controlado.
      </p>
    </>
  ),
  Component: Crop,
})
