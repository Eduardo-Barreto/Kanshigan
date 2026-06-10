#import "sbc-template.typ": sbc

#show: sbc.with(
  title: "Kanshigan: uma pipeline de visão computacional para extração automatizada de métricas em partidas de Sumô de Robôs autônomos",
  authors: (
    (name: "Eduardo Barreto"),
    (name: "Pedro Henrique de Azeredo Coutinho Cruz"),
    (name: "Luan Ramos de Mello"),
    (name: "Rodrigo Mangoni Nicola"),
  ),
  affiliation: [Instituto de Tecnologia e Liderança (Inteli), São Paulo, SP, Brasil],
  abstract: [
    Robot Sumo matches are decided in rounds that often last under one second, yet
    their analysis still relies on subjective human observation, with no automated
    tools, structured datasets, or standardized metrics. This paper presents
    Kanshigan, an open-source computer vision pipeline that detects the dohyo
    arena, tracks both robots and extracts quantitative performance metrics
    (position, velocity, acceleration) and events (round start, contact, ring-out)
    from match video. We frame the problem by the intersection of six constraints
    that no prior work covers jointly, and we build the pipeline from
    tracking-by-detection on a multi-source dataset semi-automatically annotated with
    SAM 3. We compare two detector architectures (YOLOv8s and the compact YOLO26n) and
    four trackers, two motion-only (OC-SORT, ByteTrack) and two appearance-based
    (DeepOCSORT, BoT-SORT), on held-out, manually-reviewed rounds from each
    source. Across two heterogeneous sources (handheld Brazilian and fixed-overhead
    Japanese footage), both detectors exceed mAP\@0.5 of 0.96, with no practically
    distinguishable difference between the 2.4M-parameter YOLO26n and the
    11.1M-parameter YOLOv8s on this small evaluation set (versus 0.03 for COCO weights
    without fine-tuning); SAM 3 agrees with the human gold at F1 0.96 on the Brazilian
    source; tracking reaches
    IDF1 up to 0.94; and the full pipeline runs at 133 FPS with 82 MB of GPU memory
    allocated by the detector on a consumer laptop GPU, confirming practical viability.
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
    tracking-by-detection sobre uma base multi-fonte anotada de forma semiautomática com
    o SAM 3. Comparamos duas arquiteturas de detector (YOLOv8s e o compacto YOLO26n) e
    quatro rastreadores, dois motion-only (OC-SORT, ByteTrack) e dois com aparência
    (DeepOCSORT, BoT-SORT), em rounds held-out revisados manualmente de
    cada fonte. Entre duas fontes heterogêneas (footage brasileiro de câmera de mão e
    footage japonês de câmera cenital fixa), os dois detectores superam mAP\@0.5 de 0.96,
    sem diferença praticamente distinguível entre o YOLO26n de 2,4 M de parâmetros e o
    YOLOv8s de 11,1 M neste conjunto pequeno de avaliação (contra 0.03 dos
    pesos COCO sem fine-tuning); o SAM 3 concorda com o gold humano em F1 0.96 na
    fonte brasileira; o
    rastreamento atinge IDF1 de até 0.94; e a pipeline completa roda a 133 FPS com
    82 MB de memória de GPU alocados pelo detector em uma GPU de notebook, confirmando
    viabilidade prática.
  ],
)

#include "sections/introduction.typ"
#include "sections/related-work.typ"
#include "sections/methodology.typ"
#include "sections/results.typ"
#include "sections/discussion.typ"

#bibliography("refs.bib", title: "Referências", style: "ieee")
