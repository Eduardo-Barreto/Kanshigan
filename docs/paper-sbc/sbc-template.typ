// Minimal SBC-style template (Sociedade Brasileira de Computação short paper).
// Approximates the official layout: A4, Times-compatible serif, centered title,
// English abstract + Portuguese resumo, numbered sections. Kept self-contained so
// the CI builds it with the typst toolchain already used by the other documents.

#let sbc(
  title: "",
  authors: (),
  affiliation: [],
  abstract: [],
  resumo: [],
  body,
) = {
  set document(title: title, author: authors.map(a => a.name))
  set page(paper: "a4", margin: (left: 3cm, right: 3cm, top: 3.5cm, bottom: 2.5cm))
  set text(font: ("Times New Roman", "Liberation Serif", "Libertinus Serif"), size: 12pt, lang: "pt")
  set par(justify: true, leading: 0.65em, first-line-indent: 1.25em)

  set heading(numbering: "1.1.")
  // Emit the heading as a block so it breaks from the surrounding paragraph and
  // gets its own spacing; returning inline content would glue it to the previous text.
  show heading: it => block(above: 1.2em, below: 0.8em, {
    set text(size: 12pt, weight: "bold")
    let number = counter(heading).display()
    if it.level == 1 {
      number + " " + smallcaps(it.body)
    } else {
      number + " " + it.body
    }
  })

  // Title block
  align(center)[
    #block(text(size: 16pt, weight: "bold")[#title])
    #v(0.6em)
    #block(text(size: 12pt)[#authors.map(a => a.name).join(", ", last: " and ")])
    #v(0.2em)
    #block(text(size: 11pt)[#affiliation])
  ]
  v(1.2em)

  // Abstracts
  set par(first-line-indent: 0pt)
  block(width: 100%, inset: (left: 1cm, right: 1cm))[
    #align(center)[*Abstract*]
    #v(0.3em)
    #text(size: 11pt)[#abstract]
    #v(0.8em)
    #align(center)[*Resumo*]
    #v(0.3em)
    #text(size: 11pt)[#resumo]
  ]
  v(1em)
  set par(first-line-indent: 1.25em)

  body
}
