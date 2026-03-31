#import "utils/style.typ": configure-document
#import "utils/title.typ": make-title-page

#show: configure-document.with(
  title: "Kanshigan: Project Plan",
  author: "Eduardo Barreto, Pedro Henrique Cruz, Luan Ramos de Mello"
)

#make-title-page(
  project-title: "KANSHIGAN: A COMPUTER VISION PIPELINE FOR TRACKING AND PERFORMANCE ANALYSIS IN AUTONOMOUS ROBOT SUMO COMPETITIONS",
  document-type: "Project Plan",
  track: "Academic Track – 2026",
  student-name: "Eduardo Barreto, Pedro Henrique Cruz, Luan Ramos de Mello",
  supervisor-name: "Prof. Rodrigo Mangoni Nicola"
)

#include "sections/context.typ"
#include "sections/research-problem.typ"
#include "sections/research-question.typ"
#include "sections/justification.typ"
#include "sections/relevance.typ"
#include "sections/objectives.typ"
#include "sections/methodology.typ"
#include "sections/timeline.typ"
#include "sections/references.typ"
