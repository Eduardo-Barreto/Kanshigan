#set page(paper: "a4", margin: (top: 3cm, bottom: 2cm, left: 3cm, right: 2cm))
#set text(font: "TeX Gyre Termes", size: 12pt, lang: "pt")
#set par(justify: true, leading: 1.5em, first-line-indent: 1.25cm)
#set heading(numbering: "1.1")

#show heading.where(level: 1): it => {
  v(1em)
  text(size: 14pt, weight: "bold", upper(it))
  v(0.5em)
}

#show heading.where(level: 2): it => {
  v(0.5em)
  text(size: 12pt, weight: "bold", it)
  v(0.3em)
}

#set bibliography(style: "associacao-brasileira-de-normas-tecnicas")

// Capa
#page[
  #align(center)[
    #text(size: 14pt, weight: "bold")[INSTITUTO DE TECNOLOGIA E LIDERANÇA]
    #v(0.5em)
    #text(size: 12pt)[Engenharia da Computação]

    #v(1fr)

    #text(size: 14pt, weight: "bold")[
      KANSHIGAN: UM PIPELINE DE VISÃO COMPUTACIONAL PARA RASTREAMENTO E ANÁLISE DE DESEMPENHO EM COMPETIÇÕES DE SUMÔ DE ROBÔS AUTÔNOMOS
    ]

    #v(1fr)

    #text(size: 12pt)[
      Eduardo Barreto \
      Pedro Henrique de Azeredo Coutinho Cruz \
      Luan Ramos de Mello
    ]
    #v(1em)
    #text(size: 12pt)[Orientador: Prof. Rodrigo Mangoni Nicola]

    #v(1fr)

    #text(size: 12pt)[São Paulo \ 2026]
  ]
]

#pagebreak()
#outline(title: "Sumário", indent: auto)
#pagebreak()

#include "sections/introduction.typ"
#include "sections/justification.typ"
#include "sections/objectives.typ"
#include "sections/literature-review.typ"
#include "sections/methodology.typ"
#include "sections/timeline.typ"

#bibliography("refs.bib")
