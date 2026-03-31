#import "style.typ": INTELI_COLOR

#let make-title-page(
  project-title: "PROJECT TITLE IN BOLD – SPECIFIC AND TECHNICAL",
  document-type: "Project Plan",
  institution: "Instituto de Tecnologia e Liderança – Inteli",
  track: "Academic Track – 2026",
  student-name: "[Student Name]",
  supervisor-name: "[Supervisor Name]"
) = [
  #v(120pt)
  #text(size: 14pt, weight: "bold", fill: INTELI_COLOR)[#project-title]
  #v(20pt)
  #text(size: 14pt, weight: "bold", fill: INTELI_COLOR)[#document-type]
  #v(16pt)
  #text(size: 11pt, weight: "bold", fill: INTELI_COLOR)[#institution]
  #v(12pt)

  #pad(left: 3pt)[
    #text(size: 11pt, weight: "bold", fill: INTELI_COLOR)[#track]
    #v(12pt)
    #text(size: 11pt, weight: "bold", fill: INTELI_COLOR)[Students: #student-name]
    #v(12pt)
    #text(size: 11pt, weight: "bold", fill: INTELI_COLOR)[Supervisor: #supervisor-name]
  ]

  #pagebreak()
  #v(6pt)
]
