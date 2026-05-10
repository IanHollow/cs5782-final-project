// Local project copy of @preview/charged-ieee:0.1.4, adapted for the
// CS 4782 / CS 5782 two-page report requirements.
//
// Original template: https://typst.app/universe/package/charged-ieee/
// License: MIT. See the upstream package for the original source.

#let ieee(
  title: [Paper Title],
  authors: (),
  abstract: none,
  index-terms: (),
  paper-size: "us-letter",
  bibliography: none,
  figure-supplement: [Fig.],
  body-size: 11pt,
  body-leading: 0.42em,
  paragraph-spacing: 0.42em,
  heading-size: 11pt,
  abstract-size: 10pt,
  caption-size: 9pt,
  reference-size: 9pt,
  title-size: 17pt,
  author-name-size: 9.5pt,
  author-detail-size: 8.5pt,
  author-leading: 0.36em,
  title-clearance: 14pt,
  figure-spacing: 7pt,
  table-text-size: 9pt,
  body,
) = {
  set document(title: title, author: authors.map(author => author.name))

  set text(font: "TeX Gyre Termes", size: body-size, spacing: .20em)
  set enum(numbering: "1)a)i)")

  show figure: set block(spacing: figure-spacing)
  show figure: set place(clearance: figure-spacing)
  show figure.where(kind: table): set figure.caption(
    position: top,
    separator: [\ ],
  )
  show figure.where(kind: table): set text(size: table-text-size)
  show figure.where(kind: table): set figure(numbering: "I")
  show figure.where(kind: image): set figure(
    supplement: figure-supplement,
    numbering: "1",
  )
  show figure.caption: set text(size: caption-size)
  show figure.caption: set align(start)
  show figure.caption.where(kind: table): set align(center)

  set figure.caption(separator: [. ])
  show figure: fig => {
    let prefix = (
      if fig.kind == table [TABLE] else if fig.kind
        == image [Fig.] else [#fig.supplement]
    )
    let numbers = numbering(fig.numbering, ..fig.counter.at(fig.location()))
    show figure.caption: it => block[#prefix~#numbers#it.separator#it.body]
    show figure.caption.where(kind: table): smallcaps
    fig
  }

  show raw: set text(
    font: "TeX Gyre Cursor",
    ligatures: false,
    size: 0.92em,
    spacing: 100%,
  )

  set columns(gutter: 12pt)
  set page(
    columns: 2,
    paper: paper-size,
    margin: if paper-size == "a4" {
      (x: 41.5pt, top: 70pt, bottom: 74pt)
    } else {
      (
        x: (50pt / 216mm) * 100%,
        top: 44pt,
        bottom: 50pt,
      )
    },
  )

  set math.equation(numbering: "(1)")
  show math.equation: set block(spacing: 0.55em)
  show ref: it => {
    if it.element != none and it.element.func() == math.equation {
      link(it.element.location(), numbering(
        it.element.numbering,
        ..counter(math.equation).at(it.element.location()),
      ))
    } else {
      it
    }
  }

  set enum(indent: 10pt, body-indent: 9pt)
  set list(indent: 10pt, body-indent: 9pt)

  set heading(numbering: "I.A.a)")
  show heading: it => {
    let levels = counter(heading).get()
    let deepest = if levels != () {
      levels.last()
    } else {
      1
    }

    set text(body-size, weight: 400)
    if it.level == 1 {
      let is-ack = (
        it.body
          in (
            [Acknowledgment],
            [Acknowledgement],
            [Acknowledgments],
            [Acknowledgements],
          )
      )
      set align(center)
      set text(if is-ack { body-size } else { heading-size })
      show: block.with(above: 8pt, below: 6pt, sticky: true)
      show: smallcaps
      if it.numbering != none and not is-ack {
        numbering("I.", deepest)
        h(6pt, weak: true)
      }
      it.body
    } else if it.level == 2 {
      set text(style: "italic")
      show: block.with(spacing: 6pt, sticky: true)
      if it.numbering != none {
        numbering("A.", deepest)
        h(6pt, weak: true)
      }
      it.body
    } else [
      #if it.level == 3 {
        numbering("a)", deepest)
        [ ]
      }
      _#(it.body):_
    ]
  }

  show std.bibliography: set text(reference-size)
  show std.bibliography: set block(spacing: 0.35em)
  set std.bibliography(title: text(heading-size)[References], style: "ieee")

  place(
    top,
    float: true,
    scope: "parent",
    clearance: title-clearance,
    {
      {
        set align(center)
        set par(leading: 0.38em)
        set text(size: title-size)
        block(below: 6pt, title)
      }

      set par(leading: author-leading)
      for i in range(calc.ceil(authors.len() / 4)) {
        let end = calc.min((i + 1) * 4, authors.len())
        let is-last = authors.len() == end
        let slice = authors.slice(i * 4, end)
        grid(
          columns: slice.len() * (1fr,),
          gutter: 7pt,
          ..slice.map(author => align(center, {
            text(size: author-name-size, author.name)
            if "department" in author [
              \ #emph(text(size: author-detail-size, author.department))
            ]
            if "organization" in author [
              \ #emph(text(size: author-detail-size, author.organization))
            ]
            if "location" in author [
              \ #text(size: author-detail-size, author.location)
            ]
            if "email" in author {
              if type(author.email) == str [
                \ #link("mailto:" + author.email)[#text(
                  size: author-detail-size,
                  author.email,
                )]
              ] else [
                \ #text(size: author-detail-size, author.email)
              ]
            }
          })),
        )

        if not is-last {
          v(6pt, weak: true)
        }
      }
    },
  )

  set par(
    justify: true,
    first-line-indent: 0pt,
    spacing: paragraph-spacing,
    leading: body-leading,
  )

  if abstract != none {
    set par(spacing: 0.34em, leading: 0.36em)
    set text(abstract-size, weight: 700, spacing: 115%)

    [_Abstract_---#h(weak: true, 0pt)#abstract]

    if index-terms != () {
      parbreak()
      [_Index Terms_---#h(weak: true, 0pt)#index-terms.join[, ]]
    }
    v(1pt)
  }

  body
  bibliography
}
