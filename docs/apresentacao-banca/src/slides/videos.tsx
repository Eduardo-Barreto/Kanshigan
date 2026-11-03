import './videos.css'
import { slide } from '@nimbus/deck'

import atenaBr from '../assets/videos/atena_br.mp4'
import demoJp from '../assets/videos/demo_jp.mp4'
import overlayBr from '../assets/videos/pipeline_overlay_br.mp4'
import rcSumo from '../assets/videos/rc_sumo.mp4'
import { Clip } from '../clip'

const CLIPS = [
  { src: overlayBr, label: 'Brasil · câmera de mão' },
  { src: demoJp, label: 'Japão · câmera fixa de cima' },
  { src: atenaBr, label: 'Outra arena brasileira' },
  { src: rcSumo, label: 'Sumô RC · categoria fora do treino' },
]

function Videos() {
  return (
    <div className="page videos">
      <p className="eyebrow">Em vídeo real</p>
      <h2 className="headline">O mesmo modelo, sem retreinar.</h2>
      <div className="videos-grid">
        {CLIPS.map((clip) => (
          <div key={clip.label} className="frame videos-frame">
            <Clip src={clip.src} label={`Pipeline rodando: ${clip.label}`} />
            <span className="chip">{clip.label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export const videos = slide({
  id: 'videos',
  title: 'Em vídeo real',
  notes: (
    <p>
      (35s) Isso é a pipeline rodando. Quatro contextos, o mesmo modelo, nenhum retreino. Brasil com
      câmera de mão. Japão visto de cima. Outra arena brasileira, com resolução mais baixa. E Sumô
      de rádio controle, uma categoria que nem estava no treino, com robô de outro formato e vídeo
      de celular. Roda em todos. Quão bem, olhar vídeo não diz, e aí vem a próxima medida.
    </p>
  ),
  Component: Videos,
})
