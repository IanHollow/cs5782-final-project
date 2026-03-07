#import "@preview/zebraw:0.6.1": *
#show: zebraw.with(
  ..zebraw-themes.zebra,
  numbering-separator: true,
  lang: true,
  indentation: 4,
  background-color: luma(250),
  highlight-color: blue.lighten(90%),
  comment-color: yellow.lighten(90%),
)

#let assignment = "Final Project Proposal"
#let due = datetime(day: 7, month: 3, year: 2026, hour: 23, minute: 59, second: 59)

#let course = "CS 5782 / CS 4782: Intro to Deep Learning"
#let student = (
  name: "Ian Holloway",
  netid: "imh39",
  email: "imh39@cornell.edu",
  department: "MEng, Computer Science",
  organization: "Cornell University",
  location: "Ithaca, NY, USA",
)
#let student2 = (
  name: "",
  netid: "",
  email: "@cornell.edu",
  department: "MEng, Computer Science",
  organization: "Cornell University",
  location: "Ithaca, NY, USA",
)
#let student3 = (
  name: "",
  netid: "",
  email: "@cornell.edu",
  department: "MEng, Computer Science",
  organization: "Cornell University",
  location: "Ithaca, NY, USA",
)
#let student4 = (
  name: "",
  netid: "",
  email: "@cornell.edu",
  department: "MEng, Computer Science",
  organization: "Cornell University",
  location: "Ithaca, NY, USA",
)
#let authors = (student, student2, student3, student4)

#let semester-term = dt => {
  let m = dt.month()
  if m <= 5 { "Spring" } else if m <= 7 { "Summer" } else { "Fall" }
}
#let semester-display = dt => [
  #semester-term(dt) #dt.display("[year]")
]
#let semester = semester-display(due)

#set text(
  lang: "en",
  // font: "Times New Roman",
  size: 12pt,
  top-edge: 0.8em,
  bottom-edge: -0.2em,
)
#set par(
  // first-line-indent: (
  //   amount: 2em,
  //   all: true,
  // ),
  leading: 0.5em,
  spacing: 2em,
  justify: false,
)
#let author_block(a) = align(center)[

  #strong[#a.name (#a.netid)] \
  #a.department \
  #a.organization \
  #a.location \
  #link("mailto:" + a.email)
]
#let ieee_authors(authors, gutter: 18pt) = grid(
  if authors.len() == 4 {
    stack(
      dir: ttb,
      spacing: gutter,
      grid(
        columns: (1fr,) * 3,
        gutter: gutter,
        align: center,
        ..authors.slice(0, 3).map(author_block),
      ),
      grid(
        columns: (1fr,) * 3,
        gutter: gutter,
        align: center,
        [], author_block(authors.at(3)), [],
      ),
    )
  } else {
    grid(
      columns: (1fr,) * 3,
      gutter: gutter,
      align: center,
      ..authors.map(author_block),
    )
  },
)


#set document(
  title: [#assignment --- #course],
  author: student.name,
  date: due,
)
#set page(
  paper: "us-letter",
  // margin: (x: 1in, y: 1in),
  margin: (x: 0.5in, y: 0.65in),
  header: context {
    if counter(page).get().first() > 1 [
      #stack(
        dir: ttb,
        spacing: 0.5em,
        grid(
          columns: (1fr, 1fr),
          column-gutter: 12pt,
          [#align(left)[#smallcaps[#course]]], [#align(right)[#assignment]],
        ),
        line(length: 100%, stroke: 0.5pt),
      )
    ]
  },
  footer: context {
    if counter(page).get().first() > 0 [
      #stack(dir: ttb, spacing: 0.5em, line(length: 100%, stroke: 0.5pt), grid(
        columns: (1fr, 1fr),
        column-gutter: 12pt,
        [#align(left)[#student.netid, #student2.netid, #student3.netid, #student3.netid]],
        [#align(right)[#counter(page).display("1")]],
      ))
    ]
  },
)
#set enum(numbering: "1.")
#set math.mat(delim: "[")
#set outline(indent: auto)

#align(center)[
  #title[#assignment]
  #smallcaps[#course] \
  #semester
]
#ieee_authors(authors)
