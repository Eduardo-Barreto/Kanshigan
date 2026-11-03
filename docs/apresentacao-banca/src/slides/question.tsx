import './question.css'
import { slide, Step } from '@nimbus/deck'

function Question() {
  return (
    <div className="page question">
      <p className="eyebrow">Pergunta de pesquisa</p>
      <p className="question-text">
        Quais combinações de detector e rastreador melhor equilibram acurácia e viabilidade prática
        para rastrear alvos <em>visualmente semelhantes</em>, em{' '}
        <em>movimento rápido e não linear</em>, a partir de vídeo não calibrado e sem marcadores?
      </p>
      <Step at={0} delay={0.3} as="p" className="caption question-note">
        <b>Viabilidade prática:</b> rodar no computador que as equipes têm, com velocidade e memória
        medidas.
      </Step>
    </div>
  )
}

export const question = slide({
  id: 'pergunta',
  title: 'Pergunta de pesquisa',
  notes: (
    <p>
      (25s) Tudo isso vira a pergunta: quais combinações de detector e rastreador melhor equilibram
      acurácia e viabilidade prática pra rastrear alvos quase idênticos, em movimento rápido e não
      linear, a partir de vídeo não calibrado e sem marcador? E viabilidade tem definição concreta:
      rodar no computador que as equipes têm, com velocidade e memória medidas.
    </p>
  ),
  Component: Question,
})
