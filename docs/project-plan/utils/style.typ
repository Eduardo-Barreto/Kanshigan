#import "@preview/fontawesome:0.1.0": *

#let INTELI_COLOR = rgb("#2e2640")
#let ACCENT_COLOR = rgb("#000000")
#let LINE_HEIGHT = 1.15
#let MARGIN_H = 3.175cm
#let MARGIN_V = 2.54cm

#let configure-document(title: "Project Plan", author: "", body) = {
  set document(title: title, author: author)

  set page(
    paper: "us-letter",
    margin: (top: MARGIN_V, bottom: MARGIN_V, left: MARGIN_H, right: MARGIN_H),
    background: [
      #place(
        top + left,
        dx: -0.145cm,
        dy: 0.318cm,
        image("../imgs/inteli-bg.png", width: 21.88cm, height: 27.64cm, fit: "cover")
      )
      #place(
        top + left,
        dx: 9.71cm,
        dy: 26.085cm,
        image("../imgs/inteli-logo-small.png", width: 2.156cm, height: 1.166cm)
      )
    ],
    footer: none
  )

  set text(
    font: "Manrope",
    size: 11pt,
    fill: INTELI_COLOR,
    weight: "regular"
  )

  set par(
    justify: true,
    leading: 0.5709em,  // 1.15 line height
    spacing: 10pt
  )

  set list(
    marker: "●",
    indent: 18pt,
    body-indent: 12pt,
    spacing: 9.36pt
  )
  show list: set block(above: 19.36pt, below: 19.36pt)

  // Heading styles
  show heading: it => {
    v(if it.level == 1 { 24pt } else { 10pt }, weak: true)
    it
  }

  show heading.where(level: 1): set block(above: 0pt, below: 10.87pt)
  show heading.where(level: 2): set block(above: 0pt, below: 9.58pt)
  show heading.where(level: 3): set block(above: 0pt, below: 7.04pt)

  show heading.where(level: 1): it => [
    #set text(size: 14pt, weight: "bold", fill: INTELI_COLOR)
    #it
  ]

  show heading.where(level: 2): it => [
    #set text(size: 13pt, weight: "bold", fill: INTELI_COLOR)
    #it
  ]

  show heading.where(level: 3): it => [
    #set text(size: 11pt, weight: "bold", fill: INTELI_COLOR)
    #it
  ]

  body
}
