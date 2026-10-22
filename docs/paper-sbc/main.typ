#import "sbc-template.typ": sbc

#show: sbc.with(
  title: "Kanshigan: detecção e rastreamento de robôs em vídeo de partidas de Sumô de Robôs autônomos",
  authors: (
    (name: "Eduardo Barreto"),
    (name: "Pedro Henrique de Azeredo Coutinho Cruz"),
    (name: "Luan Ramos de Mello"),
    (name: "Rodrigo Mangoni Nicola"),
  ),
  affiliation: [Instituto de Tecnologia e Liderança (Inteli), São Paulo, SP, Brasil],
  abstract: [
    Autonomous 3 kg Robot Sumo matches are still analyzed by eye. The domain embodies a general tracking problem: visually similar targets in fast, non-linear motion, filmed by uncalibrated cameras without markers. We ask which detector and tracker pairs best balance accuracy and practical viability. Kanshigan, an open tracking-by-detection pipeline, uses YOLO detectors fine-tuned on SAM 3 annotations of Brazilian handheld and Japanese overhead tournament video. On one hand-reviewed round per source, YOLOv8s and YOLO26n exceed 0.96 mAP\@0.5, within seed variation; domain fine-tuning, not architecture, drives accuracy. On one identity-annotated round, all eight pairs keep identity with at most one switch, while appearance-based trackers cut pipeline speed by more than half for no gain. With a motion-only tracker, the pipeline runs in real time on 1080p video on a laptop GPU and at 40 frames per second on CPU alone.
  ],
  resumo: [
    Partidas de Sumô de Robôs autônomos de 3 kg ainda são analisadas a olho. O domínio concentra um problema geral de rastreamento: alvos visualmente semelhantes, em movimento rápido e não linear, filmados por câmeras não calibradas e sem marcadores. Perguntamos quais pares de detector e rastreador equilibram melhor acurácia e viabilidade prática. O Kanshigan, uma pipeline aberta de rastreamento por detecção, usa detectores YOLO ajustados com anotações do SAM 3 em vídeo de torneios brasileiros (câmera de mão) e japoneses (câmera cenital). Em um round de referência por fonte, revisado à mão, o YOLOv8s e o YOLO26n passam de 0,96 de mAP\@0,5, dentro da variação entre sementes; o ajuste fino no domínio, não a arquitetura, decide a acurácia. Em um round com identidades, os oito pares mantêm a identidade com no máximo uma troca, e os rastreadores com aparência reduzem a velocidade a menos da metade sem ganho. Com um rastreador só de movimento, a pipeline roda em tempo real sobre vídeo 1080p numa GPU de notebook e a 40 quadros por segundo só na CPU.
  ],
)

#include "sections/introduction.typ"
#include "sections/related-work.typ"
#include "sections/methodology.typ"
#include "sections/results.typ"
#include "sections/discussion.typ"

#[
  #set text(hyphenate: false)
  #set par(justify: false)
  #bibliography("refs.bib", title: "Referências", style: "harvard-cite-them-right")
]
