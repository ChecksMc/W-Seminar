
#let seminararbeit(
  title: "Exposé",
  author: "ChecksMc",
  subject: "Kunst: Neue Medien",
  cohort: "2026/2028",
  kind: "Exposé",
  toc: true,
  declaration: true,
  ai-used: false,
  body,
) = {
  // ---------------------------------------
  // Document metadata
  // ---------------------------------------
  set document(
    title: [#title],
    author: author,
  )

  // ---------------------------------------
  // Page layout
  // ---------------------------------------
  let first-numbered-page = if toc { 3 } else { 2 }

  set page(
    paper: "a4",
    margin: (
      top: 3cm,
      bottom: 2.5cm,
      left: 3cm,
      right: 4cm,
    ),
    numbering: "1",
    footer: context {
      let n = counter(page).get().first()
      if n >= first-numbered-page {
        align(center, counter(page).display("1"))
      }
    },
  )

  // ---------------------------------------
  // Typography
  // ---------------------------------------
  set text(
    font: "Libertinus Serif",
    size: 12pt,
    lang: "de",
    fill: black,
    top-edge: 0.8em,
    bottom-edge: -0.2em,
  )

  // 18 pt baseline distance at 12 pt size
  set par(
    justify: true,
    leading: 0.5em,
    spacing: 0.6em,
  )

  // ---------------------------------------
  // Headings
  // ---------------------------------------
  set heading(numbering: "1.")

  show heading.where(level: 1): set text(
    size: 12pt,
    weight: "bold",
  )

  show heading.where(level: 2): set text(
    size: 12pt,
    weight: "bold",
  )

  show heading.where(level: 3): set text(
    size: 12pt,
    weight: "medium",
  )

  // ---------------------------------------
  // Title page
  // ---------------------------------------
  align(center)[
    #text(weight: "bold")[GYMNASIUM OTTOBRUNN]

    #v(1.2cm)
    Oberstufenjahrgang #cohort

    #v(1.2cm)
    Seminarfach #subject

    #v(1.2cm)
    #kind

    #v(1.2cm)
    #text(size: 15pt, weight: "bold")[#title]
  ]

  v(1.8cm)

  grid(
    columns: (3.3cm, 1fr),
    row-gutter: 0.5cm,
    [Verfasser/in:], [#author],
  )

  // Only the final paper needs a grading table.
  if kind == "Seminararbeit" {
    v(1.0cm)

    text(size: 8pt)[
      #table(
        columns: (
          2.8cm, 1.1cm, 3.2cm,
          2.1cm, 0.7cm, 1.5cm,
        ),
        stroke: 0.5pt,
        inset: 3pt,

        table.header(
          [*Bewertung*],
          [*Note*],
          [*Notenstufe in Worten*],
          [*Notenpunkte*],
          [],
          [*Punkte*],
        ),

        [Schriftliche Arbeit],
        [], [], [], [× 2], [],

        [Prüfungsgespräch],
        [], [], [], [× 1], [],

        table.cell(colspan: 5, align: right)[Summe:],
        [],

        table.cell(colspan: 5, align: right)[
          Gesamtleistung nach § 29 Abs. 6 GSO
          (Summe · 2/3, gerundet):
        ],
        [],
      )
    ]

    v(1.0cm)

    text(size: 9pt)[
      Unterschrift der Seminarleitung:
      ................................................
    ]
  }

  // ---------------------------------------
  // Table of contents
  // ---------------------------------------
  pagebreak()

  if toc {
    outline(
      title: [Inhaltsverzeichnis],
      depth: 3,
    )
    pagebreak()
  }

  // ---------------------------------------
  // Main document
  // ---------------------------------------
  body

  // ---------------------------------------
  // Final declaration
  // ---------------------------------------
  if declaration {
    pagebreak()

    heading(
      numbering: none,
      outlined: false,
    )[SCHLUSSERKLÄRUNG]

    [
      Ich versichere, dass ich die vorgelegte
      Seminararbeit persönlich und unverfälscht
      verfasst, sämtliche hierfür zu Hilfe
      genommenen gedruckten sowie digitalen
      Quellen im Literaturverzeichnis angegeben
      und die aus diesen Quellen stammenden
      Zitate oder Belegstellen für sinngemäß
      wiedergegebene Inhalte in meiner
      Seminararbeit als solche kenntlich
      gemacht habe.
    ]

    if ai-used [
      Bei der Anfertigung der vorliegenden
      Arbeit habe ich gemäß der oben
      angegebenen Tabelle KI verwendet.
      Nach dem Einsatz habe ich die generierten
      Ergebnisse jeweils vollumfänglich
      geprüft und, soweit erforderlich,
      überarbeitet.
    ]

    [
      Ich versichere die lückenlose Einhaltung
      der erlernten wissenschaftlichen Standards
      und übernehme die volle Verantwortung
      für die gesamte vorliegende Arbeit.

      Die Seminararbeit ist in dieser oder
      einer ähnlichen Form in keinem anderen
      Kurs des diesjährigen oder eines
      vorhergehenden Abiturjahrgangs
      vorgelegt worden.
    ]

    v(1.5cm)

    grid(
      columns: (1fr, 1fr),
      column-gutter: 1cm,
      [................................],
      [................................],
      [Ort, Datum],
      [Unterschrift],
    )
  }
}
