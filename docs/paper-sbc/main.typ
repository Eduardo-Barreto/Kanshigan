#import "sbc-template.typ": sbc

#show: sbc.with(
  title: "Kanshigan: uma pipeline de visão computacional para extração automatizada de métricas em partidas de Sumô de Robôs autônomos",
  authors: (
    (name: "Eduardo Barreto"),
    (name: "Pedro Henrique de Azeredo Coutinho Cruz"),
    (name: "Luan Ramos de Mello"),
    (name: "Rodrigo Mangoni Nicola"),
  ),
  affiliation: [Instituto de Tecnologia e Liderança (Inteli) --- São Paulo, SP --- Brasil],
  abstract: [
    Robot Sumo matches are decided in rounds that often last under one second, yet
    their analysis still relies on subjective human observation, with no automated
    tools, structured datasets, or standardized metrics. This paper presents
    Kanshigan, an open-source computer vision pipeline that detects the dohyo
    arena, tracks both robots and extracts quantitative performance metrics
    (position, velocity, acceleration) and events (round start, contact, ring-out)
    from match video. We frame the problem by the intersection of six constraints
    that no prior work covers jointly, and we build the pipeline from
    tracking-by-detection: a YOLOv8s detector trained on a dataset semi-automatically
    annotated with SAM 3, followed by OC-SORT tracking. On real tournament footage,
    the detector reaches mAP\@0.5 of 0.98 with 0.98 recall on a held-out,
    manually-reviewed round (versus 0.03 for COCO weights without fine-tuning), and
    the full pipeline runs at 133 FPS using 82 MB of GPU memory on a consumer laptop
    GPU, confirming practical viability.
  ],
  resumo: [
    Partidas de Sumô de Robôs são decididas em rounds que frequentemente duram
    menos de um segundo, mas sua análise ainda depende de observação humana
    subjetiva, sem ferramentas automatizadas, bases de dados estruturadas ou
    métricas padronizadas. Este artigo apresenta o Kanshigan, uma pipeline de
    visão computacional de código aberto que detecta a arena (dohyo), rastreia os
    dois robôs e extrai métricas quantitativas de desempenho (posição, velocidade,
    aceleração) e eventos (início do round, contato, ring-out) a partir do vídeo da
    partida. Caracterizamos o problema pela interseção de seis restrições que
    nenhum trabalho anterior cobre em conjunto, e construímos a pipeline a partir do
    paradigma tracking-by-detection: um detector YOLOv8s treinado sobre uma base
    anotada de forma semiautomática com o SAM 3, seguido de rastreamento OC-SORT.
    Sobre footage real de torneios, o detector atinge mAP\@0.5 de 0.98 com recall de
    0.98 em um round held-out revisado manualmente (contra 0.03 dos pesos COCO sem
    fine-tuning), e a pipeline completa roda a 133 FPS usando 82 MB de memória de GPU
    em uma GPU de notebook de consumo, confirmando viabilidade prática.
  ],
)

#include "sections/introduction.typ"
#include "sections/related-work.typ"
#include "sections/methodology.typ"
#include "sections/results.typ"
#include "sections/discussion.typ"

#bibliography("refs.bib", title: "Referências", style: "ieee")
