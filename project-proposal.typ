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
  name: "Younghyun Jung",
  netid: "yj582",
  email: "@cornell.edu",
  department: "MEng, Computer Science",
  organization: "Cornell University",
  location: "Ithaca, NY, USA",
)
#let student3 = (
  name: "Swathi Saravana Selvam",
  netid: "",
  email: "@cornell.edu",
  department: "MEng, Computer Science",
  organization: "Cornell University",
  location: "Ithaca, NY, USA",
)
#let student4 = (
  name: "Shashwat Modi",
  netid: "",
  email: "@cornell.edu",
  department: "",
  organization: "Cornell University",
  location: "Ithaca, NY, USA",
)
#let authors = (student, student2, student3, student4)
#let active_authors = authors.filter(a => a.name != "")

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
#let ieee_authors(authors, gutter: 18pt) = {
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
  } else if authors.len() == 1 {
    align(center)[#author_block(authors.at(0))]
  } else {
    grid(
      columns: (1fr,) * 3,
      gutter: gutter,
      align: center,
      ..authors.map(author_block),
    )
  }
}
#let footer_netids = if active_authors.len() > 0 {
  active_authors.map(a => a.netid).join(", ")
} else {
  ""
}


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
        [#align(left)[#footer_netids]], [#align(right)[#counter(page).display("1")]],
      ))
    ]
  },
)
#set enum(numbering: "1.")
#set math.mat(delim: "[")
#set outline(indent: auto)

#align(center)[
  #title[#assignment]
  #strong[Reproducing DoRA for Commonsense Reasoning with Student-Scale Compute] \
  #smallcaps[#course] \
  #semester
]
#ieee_authors(active_authors)


= Paper Selection


- #strong[Title:] #link("https://arxiv.org/abs/2402.09353")[*DoRA: Weight-Decomposed Low-Rank Adaptation*]

- #strong[Authors:] Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen

- #strong[Venue:] ICML 2024 Oral

- #strong[Brief summary:] DoRA is a parameter-efficient fine-tuning method based on LoRA. Instead of learning only a low-rank update, it separates each pretrained weight into a magnitude and a direction part. The method keeps low-rank updates for the direction and learns the magnitude separately. The paper shows that this usually improves accuracy over standard LoRA.

- #strong[Why we chose this paper:] The paper has a clear main claim, official code is available, the benchmark tasks are standard, and the project is challenging but still feasible for a student team with Colab Pro.


= Data and Availability


- The fine-tuning dataset is `commonsense_170k.json` from the LLM-Adapters repository.

- The evaluation tasks are BoolQ, PIQA, Social IQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, and OpenBookQA.

- The official NVlabs DoRA commonsense reasoning README directly links to these resources and provides evaluation scripts for them.

- This means the data exists, is publicly accessible, and is already supported by the official reproduction pipeline.


= Result Selection


- We want to replicate the main commonsense reasoning result from the paper: DoRA should outperform LoRA on average across the 8 benchmark tasks.

- Our main target is the average accuracy improvement of DoRA over LoRA under a matched setup.

- Our preferred setup is the official LLaMA-7B-style commonsense experiment.

- If model access or runtime limits make that difficult, we will use the official Llama-2-7B or Llama-3-8B commonsense setup instead.

- Success means reproducing the main trend of the paper and getting reasonably close results, even if we do not match every exact number.


= Re-implementation Plan


- #strong[Architecture and method:] We will fine-tune a frozen Llama-family base model with LoRA and DoRA adapters and compare them under the same setup. We will follow the official [NVlabs commonsense reasoning scripts](https://github.com/NVlabs/DoRA/tree/main/commonsense_reasoning). DoRA will be applied to `q_proj`, `k_proj`, `v_proj`, `up_proj`, and `down_proj`.

- #strong[Models:] Our first choice is the default LLaMA-7B setting. If access is blocked, we will use the officially released Llama-2-7B or Llama-3-8B setup.

- #strong[Metrics:] We will report accuracy for each of the 8 tasks and the macro average across all tasks.

- #strong[Main settings:] We plan to use the released hyperparameters as closely as possible, including 3 training epochs, cutoff length 256, batch size 16, gradient checkpointing, and the official learning rates for the chosen model.

- #strong[Tools:] We will use the official NVlabs DoRA code when possible, Hugging Face `transformers`, and Hugging Face `peft` with DoRA support. Training and evaluation will be run mainly on Google Colab Pro, while the M4 MacBook Pro will be used for debugging, plotting, and organizing results.

- #strong[Compute and time:] Since all four team members have Google Colab Pro, the project is feasible. A main 7B or 8B fine-tuning run should take about half a day to one day, depending on the GPU assigned by Colab. Full evaluation should take a few hours. This is realistic for our team because we can split runs and analysis across four Colab Pro accounts.



