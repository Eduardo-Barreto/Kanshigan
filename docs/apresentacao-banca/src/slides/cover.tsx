import './cover.css'
import { slide, Step } from '@nimbus/deck'

import overlay from '../assets/figures/overlay_frame.webp'

function Cover() {
  return (
    <div className="page cover">
      <div className="cover-text">
        <p className="eyebrow">Trabalho de conclusão · Inteli</p>
        <h1 className="cover-title">Kanshigan</h1>
        <p className="lede cover-lede">
          Uma pipeline aberta que detecta e rastreia robôs de Sumô a partir do vídeo da luta.
        </p>
        <dl className="cover-people">
          <div>
            <dt>Autores</dt>
            <dd>Eduardo Barreto · Pedro Cruz · Luan Ramos</dd>
          </div>
          <div>
            <dt>Orientador</dt>
            <dd>Rodrigo Mangoni Nicola</dd>
          </div>
        </dl>
      </div>
      <Step motion="pop" delay={0.1} className="frame cover-frame">
        <img
          src={overlay}
          alt="Saída da pipeline: arena em amarelo, robô A em verde e robô B em laranja"
        />
        <span className="chip">Saída da pipeline · IRONCup 2025</span>
      </Step>
    </div>
  )
}

export const cover = slide({
  id: 'capa',
  title: 'Capa',
  notes: (
    <p>
      (20s) Bom dia. Eu sou o Eduardo, e esse é o Kanshigan, nosso TCC: uma pipeline de visão
      computacional que mede partidas de Sumô de Robôs a partir do vídeo da luta. Essa imagem já é a
      saída dela: a arena em amarelo, o robô A em verde e o B em laranja.
    </p>
  ),
  Component: Cover,
})
