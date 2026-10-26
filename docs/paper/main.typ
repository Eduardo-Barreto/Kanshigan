#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Kanshigan: Detecting and Tracking Robots in Autonomous Robot Sumo Match Video],
  abstract: [
    Autonomous 3 kg Robot Sumo matches are still analyzed by eye. The domain embodies a general tracking problem: visually similar targets in fast, non-linear motion, filmed by uncalibrated cameras without markers. We ask which detector and tracker pairs best balance accuracy and practical viability in this regime. Kanshigan, an open tracking-by-detection pipeline, uses YOLO detectors fine-tuned on SAM 3 annotations of Brazilian handheld and Japanese overhead tournament video. On one hand-reviewed round per source, YOLOv8s and YOLO26n exceed 0.96 mAP\@0.5, within seed variation; domain fine-tuning, not architecture, drives accuracy (from 0.36 to above 0.97 on the overhead view). On one identity-annotated round, all eight detector and tracker pairs keep identity with at most one switch, while appearance-based trackers cut pipeline speed by more than half for no gain. Measured directly, the two robots in that round are separable by appearance; motion alone already preserves identity. With a motion-only tracker, the full pipeline runs in real time on 1080p video on a laptop GPU and at 40 frames per second on CPU alone. A label-free audit anchored on the reference round reveals a failure mode: extreme motion blur in broadcast footage.
  ],
  authors: (
    (
      name: "Eduardo Barreto",
      organization: [Instituto de Tecnologia e Liderança],
      location: [São Paulo, Brasil],
      email: "eu@barreto.sh"
    ),
    (
      name: "Pedro Henrique de Azeredo Coutinho Cruz",
      organization: [Instituto de Tecnologia e Liderança],
      location: [São Paulo, Brasil],
      email: "pedro.cruz@sou.inteli.edu.br"
    ),
    (
      name: "Luan Ramos de Mello",
      organization: [Instituto de Tecnologia e Liderança],
      location: [São Paulo, Brasil],
      email: "luan.mello@sou.inteli.edu.br"
    ),
    (
      name: "Rodrigo Mangoni Nicola",
      organization: [Instituto de Tecnologia e Liderança],
      location: [São Paulo, Brasil],
      email: "rodrigo.nicola0@gmail.com"
    ),
  ),
  index-terms: (
    "Robot Sumo",
    "Multi-Object Tracking",
    "Object Detection",
    "Foundation-Model Annotation",
    "Sports Analytics",
  ),
  bibliography: bibliography("refs.bib"),
)

#include "sections/introduction.typ"
#include "sections/related-work.typ"
#include "sections/methodology.typ"
#include "sections/results.typ"
#include "sections/discussion.typ"
