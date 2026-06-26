# Metodologia explícita e alegação de ineditismo defensável

## Contexto

A banca confrontou dois pontos do artigo SBC. Primeiro, a metodologia descrevia a
pipeline estágio a estágio, mas nunca nomeava QUAL metodologia de pesquisa sustenta o
trabalho. Segundo, a alegação de que "não existe pesquisa" foi contestada, porque visão
computacional aparece, sim, no Sumô de Robôs. Esta entrada documenta a correção dos dois.

## O que estava errado

- A seção 3 (Metodologia) entrava direto em "Detecção da arena", sem classificar a
  pesquisa nem expor o desenho experimental (variáveis manipuladas, controles, variáveis
  dependentes, protocolo). A classificação aplicada/quantitativa/exploratória/experimental
  só existia na `projeto-de-pesquisa`, nunca chegou ao paper SBC.
- Resultados e a tabela de viabilidade referenciam E1/E2/E3, mas nenhuma seção definia
  esses pontos experimentais. Referência pendente que um revisor pega.
- A alegação de ineditismo era absoluta ("Nenhum trabalho publicado trata...") e a nota
  de rodapé da introdução dizia "em bases acadêmicas", quando a busca real foi web
  indexada + arXiv (ver `docs/busca-literatura-ineditismo.md`). Overclaim no método e
  underclaim no rigor.

## O que mudou

Como bons papers de MOT/esportes fazem (MOTChallenge, DanceTrack, SportsMOT): abrir a
metodologia com o desenho experimental, não com o passo a passo do sistema.

- `methodology.typ`: nova subseção "Classificação e desenho do estudo". Nomeia a natureza
  (aplicada), abordagem (quantitativa), objetivos (exploratória) e procedimentos
  (experimental); declara os dois fatores manipulados (arquitetura do detector; algoritmo
  de rastreamento) e o controle que isola cada um (mesma divisão/semente/treino;
  detecções fixas); define E1/E2/E3, fechando as referências pendentes em resultados.
- `related-work.typ`: abertura reescrita. Concede que a visão já existe no Sumô como
  percepção embarcada (objetivo oposto), e escopa a alegação para análise post-match em
  vídeo de terceira pessoa. Nota de rodapé documenta a busca de forma honesta (web
  indexada + arXiv, junho 2026, quatro strings), sem fingir bases autenticadas.
- `introduction.typ`: claim suavizado para "até onde nossa busca alcança" e escopado a
  análise post-match; rodapé reescrito com o escopo preciso, apontando o protocolo na
  seção de trabalhos relacionados.

## Verificação

`typst compile` do artigo SBC passa limpo após as três edições.

## Pendência

A defesa mais forte do ineditismo ainda depende de rodar as strings nos portais
autenticados (IEEE Xplore, ACM DL, Scopus) e registrar contagens, como já listado em
`docs/busca-literatura-ineditismo.md`. O paper hoje só afirma o que a busca realmente
cobriu (web indexada + arXiv); rodar os portais permitiria endurecer a nota de rodapé.
</content>
</invoke>
