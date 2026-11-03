import './problem.css'
import { slide, Step } from '@nimbus/deck'

import worldsRaw from '../assets/videos/worlds_raw.mp4'
import { Clip } from '../clip'

function Problem() {
  return (
    <div className="page problem">
      <div className="frame problem-frame">
        <Clip
          src={worldsRaw}
          label="Round da final do mundial japonês, 3 kg autônomo"
          stillAt={2}
        />
        <span className="chip">Final do All Japan Robot Sumo · 3 kg autônomo</span>
      </div>
      <div className="problem-text">
        <h2 className="headline">Do primeiro contato à saída da arena, cerca de um segundo.</h2>
        <Step at={1} as="p" className="lede">
          E hoje tudo é analisado no olho, de memória e no replay.
          <b> Nenhuma medida de velocidade, reação ou trajetória.</b>
        </Step>
      </div>
    </div>
  )
}

export const problem = slide({
  id: 'problema',
  title: 'O problema',
  steps: 2,
  notes: (
    <>
      <p>
        (35s) Esse é o cenário: final do mundial japonês, categoria 3 kg autônomo. Repara na
        velocidade. [deixar o round correr] A fase que decide a luta, do primeiro contato até um
        robô sair da arena, tem mediana de 1,1 segundo em 9 rounds japoneses que a gente revisou à
        mão.
      </p>
      <p>
        [avançar] E hoje tudo isso é analisado no olho. As equipes ajustam estratégia e robô pela
        memória e pelo replay, sem medir velocidade, tempo de reação ou trajetória. Eu compito nessa
        categoria, e é assim em todo torneio.
      </p>
    </>
  ),
  Component: Problem,
})
