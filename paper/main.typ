#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Kanshigan: A Computer Vision Pipeline for Tracking and Performance Analysis in Autonomous Robot Sumo Competitions],
  abstract: [
    Robot Sumo competitions feature autonomous robots competing in ultra-fast matches where entire rounds last under one second. Currently, all performance analysis relies on subjective human observation with no automated tools, structured databases, or standardized metrics available. This paper presents Kanshigan, the first open-source computer vision pipeline for automated detection, tracking, and performance metric extraction from Robot Sumo match videos. We construct and publicly release the first annotated dataset of 3kg autonomous Robot Sumo matches, sourced from international and Brazilian competitions. We implement and compare three detection-tracking approaches --- YOLOv8+ByteTrack, YOLOv8+BoT-SORT, and RT-DETR+ByteTrack --- evaluating accuracy, robustness to heterogeneous video quality, and real-time viability. Our pipeline automatically detects the dohyo arena, tracks both robots with consistent identity assignment, extracts quantitative metrics (position, velocity, acceleration, reaction time), and identifies key events (round start, contact, ring-out). Results demonstrate that (TODO: summarize key findings). The dataset, source code, and trained models are publicly available at (TODO: link).
  ],
  authors: (
    (
      name: "Eduardo Barreto",
      organization: [Instituto de Tecnologia e Liderança],
      location: [São Paulo, Brasil],
      email: "TODO"
    ),
    (
      name: "Pedro Henrique de Azeredo Cruz",
      organization: [Instituto de Tecnologia e Liderança],
      location: [São Paulo, Brasil],
      email: "TODO"
    ),
    (
      name: "Luan Ramos de Mello",
      organization: [Instituto de Tecnologia e Liderança],
      location: [São Paulo, Brasil],
      email: "TODO"
    ),
    (
      name: "Rodrigo Mangoni Nicola",
      organization: [Instituto de Tecnologia e Liderança],
      location: [São Paulo, Brasil],
      email: "TODO"
    ),
  ),
  index-terms: (
    "Robot Sumo",
    "Computer Vision",
    "Object Tracking",
    "Sports Analytics",
    "Performance Analysis",
  ),
  bibliography: bibliography("refs.bib"),
)

#include "sections/introduction.typ"
#include "sections/related-work.typ"
#include "sections/dataset.typ"
#include "sections/methodology.typ"
#include "sections/experiments.typ"
#include "sections/conclusion.typ"
