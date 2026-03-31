#import "style.typ": INTELI_COLOR

#let section-with-guidance(heading-text, level: 1, guidance: "", content: none) = [
  #if level == 1 [
    = #heading-text
  ] else if level == 2 [
    == #heading-text
  ] else if level == 3 [
    === #heading-text
  ]

  #if guidance != "" [
    #v(12pt)
    #text(fill: INTELI_COLOR)[#guidance]
  ]

  #if content != none [
    #v(12pt)
    #content
  ]
]

#let section-with-bullets(heading-text, level: 1, bullets: ()) = [
  #if level == 1 [
    = #heading-text
  ] else if level == 2 [
    == #heading-text
  ] else if level == 3 [
    === #heading-text
  ]

  #v(12pt)
  #list(
    ..bullets
  )
]

#let section-with-subsections(heading-text, guidance: "", subsections: ()) = [
  = #heading-text

  #if guidance != "" [
    #v(12pt)
    #text(fill: INTELI_COLOR)[#guidance]
  ]

  #for subsection in subsections [
    #v(12pt)
    == #subsection.at("title")
    #v(12pt)
    #subsection.at("content")
  ]
]

// Objectives subsection (general + specific)
#let make-objectives-section(
  general-objective: "[General objective here, starting with an infinitive verb.]",
  specific-objectives: (
    "Begin with an infinitive verb",
    "Be measurable",
    "Be executable",
    "Correspond to a methodological step"
  )
) = [
  = 6 OBJECTIVES

  #v(12pt)
  == 6.1 General Objective
  #v(12pt)

  #general-objective

  #v(12pt)

  == 6.2 Specific Objectives
  #v(12pt)

  #list(
    ..specific-objectives
  )

  #v(12pt)
]

// Methodology subsection template
#let make-methodology-section(
  research-design: "[Design description]",
  data: "[Data description]",
  variables: "[Variables description]",
  tools: "[Tools description]",
  metrics: "[Metrics description]"
) = [
  = 7 METHODOLOGY

  #v(12pt)
  == 7.1 Research Design
  #v(12pt)
  #research-design
  #v(20pt)

  == 7.2 Data
  #v(3pt)
  #data
  #v(18pt)

  == 7.3 Variables
  #v(3pt)
  #variables
  #v(18pt)

  == 7.4 Tools and Infrastructure
  #v(3pt)
  #tools
  #v(18pt)

  == 7.5 Evaluation Metrics
  #v(3pt)
  #metrics
]

// Timeline section
#let make-timeline-section(
  module-name: "Module 1",
  sprints: ("Sprint 1: Description", "Sprint 2: Description")
) = [
  = 9 PROJECT TIMELINE (Aligned with Module Sprints)

  #v(12pt)

  The project development began in Minicourse 1 and will continue across subsequent modules.

  #v(12pt)

  For #module-name, the research evolved according to five sprints:

  #list(
    ..sprints
  )

  Future modules will expand:

  - Experimental execution
  - Data collection
  - Implementation
  - Evaluation
  - Scientific writing
]

// References section
#let make-references-section(
  references: ("[1] Reference here", "[2] Reference here")
) = [
  = 10 REFERENCES

  #v(12pt)
  #for ref in references [
    #ref
    #v(4pt)
  ]
]
