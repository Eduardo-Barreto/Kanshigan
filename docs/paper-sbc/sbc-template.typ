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
  set text(font: ("Liberation Serif", "Libertinus Serif"), size: 12pt, lang: "pt", region: "br")
  set par(justify: true, leading: 0.65em, first-line-indent: 1.25em)

  set heading(numbering: "1.1.")
  // Figure and table captions: smaller than body, centered, not justified, so they
  // read as captions ("Tabela 1: ...") instead of blending into the running text.
  show figure.caption: it => block(width: 100%, {
    set text(size: 10pt)
    set par(justify: false, first-line-indent: 0pt, leading: 0.55em)
    align(center, it)
  })
  show figure: set block(breakable: false)
  set figure(placement: auto)
  show figure.where(kind: table): set figure.caption(position: top)
  // Emit the heading as a block so it breaks from the surrounding paragraph and
  // gets its own spacing; returning inline content would glue it to the previous text.
  show heading: it => block(above: 1.2em, below: 0.8em, {
    set text(size: 12pt, weight: "bold")
    if it.numbering == none {
      it.body
    } else {
      counter(heading).display() + " " + it.body
    }
  })

  // Title block. Reset paragraph justification and indent here: the global
  // `justify: true` would stretch a wrapped title's first line with ragged gaps,
  // and `first-line-indent` would shove it right, which reads as a broken title.
  align(center)[
    #set par(justify: false, first-line-indent: 0pt)
    #block(text(size: 16pt, weight: "bold")[#title])
    #v(0.6em)
    #block(text(size: 12pt)[#authors.map(a => a.name).join(", ", last: " e ")])
    #v(0.2em)
    #block(text(size: 11pt)[#affiliation])
  ]
  v(1.2em)

  // Abstracts
  set par(first-line-indent: 0pt)
  block(width: 100%, inset: (left: 0.8cm, right: 0.8cm))[
    #set par(justify: true)
    #text(style: "italic", lang: "en", region: none)[*Abstract.* #abstract]
    #v(0.8em)
    #text(style: "italic")[*Resumo.* #resumo]
  ]
  v(1em)
  set par(first-line-indent: 1.25em)

  body
}
