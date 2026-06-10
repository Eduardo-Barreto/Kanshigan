# Protocolo de busca de literatura: alegação de ineditismo

Material de defesa para a banca. Registra a busca que sustenta as afirmações do
artigo de que não há, na literatura, pipeline, dataset ou benchmark publicado de
análise de partidas de Sumô de Robôs por visão computacional em terceira pessoa.

## Escopo da alegação

A alegação é específica: não encontramos trabalho publicado que faça **análise
post-match de partidas de Sumô de Robôs a partir de vídeo em terceira pessoa**
(detecção da arena, rastreamento dos dois robôs, extração de métricas cinemáticas e
detecção de eventos). A alegação **não** nega a existência de:

- percepção **embarcada** no robô (sensores IR/ultrassônicos, visão onboard para
  detectar o oponente durante a luta);
- datasets de detecção de objeto de hobby (caixas em torno de robôs);
- análise por visão de outros esportes ou de Sumô humano.

Esses casos adjacentes existem e são justamente o contraste que delimita a lacuna.

## Método

- Data da busca: 2026-06-09.
- Ferramenta: busca web indexada (Google), com restrição de domínio para arxiv.org
  em parte das consultas.
- Limitação assumida (ver pendência abaixo): esta rodada não consultou os portais
  IEEE Xplore, ACM DL e Scopus autenticados, então não reporta contagem exata de
  resultados por base. Reporta as strings, os achados e a categorização.

### Strings consultadas

1. `"robot sumo" OR "sumo robot" computer vision detection tracking match analysis`
2. `automated refereeing scoring "robot sumo" vision pipeline dataset`
3. `robot sumo match video analysis kinematics referee` (restrito a arxiv.org)
4. `"combat robot" OR "robot combat" computer vision automated analysis tracking dataset benchmark`

## Resultados, por categoria

**Percepção embarcada / detecção de oponente onboard** (não é análise de partida):

- Parallax SumoBot WX: detecção e rastreamento de oponente por IR.
- Robot Room "Number Two": varredura ultrassônica em servo para localizar o oponente.
- Deep Learning Sumo Robot (Hackster.io / Seeed Studio): modelo de deep learning
  embarcado, framework aXeleRate, para detectar o oponente.

**Datasets de detecção de objeto (hobby), não gold de análise de partida:**

- Sumo Object Detection Dataset (Kirikkale University, Roboflow Universe): caixas de
  detecção, sem rastreamento de identidade, cinemática ou eventos.
- Além de não terem identidade, cinemática nem eventos, esses datasets de hobby são
  antigos e não representam os robôs da competição atual (chassi, materiais, escala da
  categoria de 3 kg autônomo). Mesmo para a sub-tarefa de detecção, não são um
  substituto representativo do domínio que o artigo trata.

**Sumô humano (não robôs):**

- Patentes de sumarização/processamento de vídeo de Sumô humano (USPTO).

**Combate em outros contextos:**

- CombatVLA (arXiv 2503.09527): combate em jogos 3D (RPG de ação), não robôs físicos.
- Benchmarks de rastreamento de pessoas por robôs móveis (LIDAR), não análise de luta.

## Conclusão

Nenhum resultado cobre o problema-alvo. Os trabalhos de visão em Sumô de Robôs são de
percepção embarcada (o robô vendo o oponente), objetivo oposto ao nosso (um observador
externo medindo a partida). Os datasets são de detecção de objeto de hobby, sem
identidade, cinemática nem eventos. Análise de combate por visão aparece só em jogos ou
em Sumô humano. A interseção que o artigo caracteriza (arena circular, dois robôs quase
idênticos, eventos sub-segundo, vídeo heterogêneo, análise post-match, sem fiduciais)
permanece sem cobertura publicada.

## Pendência (a finalizar antes da arguição)

Para resposta robusta em banca, rodar as mesmas strings nos portais autenticados e
registrar a contagem exata:

- [ ] IEEE Xplore: strings 1 e 4, registrar nº de resultados e período.
- [ ] ACM Digital Library: idem.
- [ ] Scopus: idem, com operadores de campo (TITLE-ABS-KEY).
- [ ] arXiv (listing API): `"robot sumo"` em cs.CV e cs.RO.
- [ ] Atualizar este documento com bases, strings, datas e contagens.
