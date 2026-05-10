#import "charged-ieee.typ": ieee

#show: ieee.with(
  title: [DoRA Reimplementation for Commonsense Reasoning],
  abstract: none,
  authors: (
    (
      name: "Ian Holloway",
      department: [MEng Computer Science],
      organization: [Cornell University],
      location: [Ithaca, NY, USA],
      email: "imh39@cornell.edu",
    ),
    (
      name: "Younghyun Jung",
      department: [MEng Computer Science],
      organization: [Cornell University],
      location: [Ithaca, NY, USA],
      email: "yj582@cornell.edu",
    ),
    (
      name: "Swathi Selvam",
      department: [MEng Computer Science],
      organization: [Cornell University],
      location: [Ithaca, NY, USA],
      email: "ss4522@cornell.edu",
    ),
    (
      name: "Shashwat Modi",
      department: [MPS DS & Appl. Stats.],
      organization: [Cornell University],
      location: [Ithaca, NY, USA],
      email: "sm3225@cornell.edu",
    ),
  ),
  index-terms: (),
  paper-size: "us-letter",
  bibliography: bibliography("/report/refs.bib"),
  figure-supplement: [Figure],
  body-size: 11pt,
  body-leading: 0.40em,
  paragraph-spacing: 0.72em,
  heading-size: 11pt,
  abstract-size: 11pt,
  caption-size: 11pt,
  reference-size: 11pt,
  title-size: 17pt,
  author-name-size: 11pt,
  author-detail-size: 11pt,
  author-leading: 0.40em,
  title-clearance: 12pt,
  figure-spacing: 5pt,
  table-text-size: 11pt,
)

#show figure.caption.where(kind: image): set align(center)

#align(
  center,
)[*Code:* #link("https://github.com/IanHollow/cs5782-final-project")[github.com/IanHollow/cs5782-final-project]]

= Introduction

Full fine-tuning adapts every parameter of a large language model, making each downstream task expensive to train and store. Parameter-efficient fine-tuning (PEFT) instead freezes the base model and trains small adapter modules. Our project reimplements a central result from *DoRA: Weight-Decomposed Low-Rank Adaptation* by Liu et al. @liu2024dora, which extends LoRA @hu2022lora by separating the direction and magnitude of an adapted weight.

LoRA keeps a pretrained weight $W$ frozen and learns a low-rank residual $Delta W = B A$. DoRA keeps the low-rank directional update but learns a per-output magnitude vector $m$:

$ W' = m dot (W + Delta W) / norm(W + Delta W). $

Our contribution is a controlled student-scale reproduction with repo-owned PEFT code rather than a wrapper-based experiment. We implemented both LoRA and DoRA from scratch in PyTorch, reproduced the paper's main LoRA-vs-DoRA comparison in the same benchmark family, and added scope and rank ablations to test when DoRA helps.

= Chosen Result

We targeted Table 1 of the DoRA paper: commonsense reasoning with LLaMA-2-7B, comparing LoRA rank 32 against rank-halved DoRA rank 16 on BoolQ, PIQA, Social IQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, and OpenBookQA. This table is central to the paper because it supports DoRA's main claim: higher accuracy than LoRA with fewer trainable parameters. The official result reports macro accuracy rising from 77.6% for LoRA to 80.5% for DoRA-dagger while trainable parameters fall from 0.83% to 0.43%.

We chose this result because it is quantitative, directly tied to the paper's main contribution, and feasible to investigate under student compute constraints. Our deliberate scale change was training on a 15k commonsense subset instead of the paper's 170k training set. This smaller setting still tests the core claim because matched LoRA and DoRA runs share the same base model, data subset, prompts, benchmark files, seed, and evaluation code. We also asked whether DoRA's benefit depends on adapter placement and whether lower DoRA ranks change the accuracy-parameter tradeoff.

= Methodology

#figure(
  placement: none,
  caption: [Experimental setup. The 15k subset is the deliberate scale reduction; the model family, benchmark suite, rank-halved comparison, and metric follow the paper's commonsense setting.],
  table(
    columns: (0.25fr, 0.70fr),
    align: (left, left),
    stroke: (_, y) => if y <= 1 { (top: 0.45pt) } else { none },
    table.header[Component][Choice],
    [Model], [Meta LLaMA-2-7B, frozen base weights],
    [Training data], [Commonsense 15k instead of the paper's 170k version],

    [Benchmarks],
    [BoolQ, PIQA, Social IQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA],

    [Runtime], [Google Colab A100 with 4-bit NF4],
    [Main comp.], [LoRA rank 32 vs. DoRA rank 16],

    [Training/eval],
    [3 epochs, cutoff length 256, seed 42, adapter-only checkpoints, per-task and macro accuracy],
  ),
) <tab:setup>

The implementation uses Hugging Face Transformers only for model/tokenizer loading @transformers. The PEFT logic is repo-owned. Our LoRA adapter wraps a frozen base linear layer and adds a learned low-rank residual. Our DoRA adapter initializes a trainable per-output-channel magnitude from the pretrained weight norm, adds the low-rank update to the frozen weight, normalizes the resulting direction, and rescales it by the learned magnitude.

The system also implements in-place adapter injection after freezing the base model, adapter-only checkpoints, reloadable configs, standard linear and 4-bit quantized linear targets, benchmark evaluation, and result aggregation. We evaluated three adapter scopes: full adapters on query, key, value, up, and down projections; attention-only adapters on query, key, and value; and MLP-only adapters on up and down projections. This ablation tests whether DoRA is a local formula improvement or whether it needs broad transformer coverage.

We kept the comparison controlled: same base model, 15k subset, prompt/evaluation code, seed, benchmark files, and macro-average metric for matched LoRA and DoRA runs. Correctness is supported by 14 test files with 50 tests covering adapter math, injection, checkpointing, config loading, data processing, evaluation, and CLI behavior. The main limitations are practical: one seed, Colab A100 4-bit runtime, 15k instead of 170k training examples, and no full hyperparameter search.

= Results & Analysis

#figure(
  placement: none,
  caption: [Paper result versus our full-scope reproduction. Macro is unweighted average accuracy over the eight commonsense tasks.],
  table(
    columns: (0.4fr, 0.15fr, 0.2fr, 0.2fr),
    align: (left, right, right, right),
    stroke: (_, y) => if y <= 1 { (top: 0.45pt) } else { none },
    table.header[Condition][Rank][Macro][Trainable],
    [Paper LoRA], [32], [77.60%], [0.83%],
    [Paper DoRA-dagger], [16], [80.50%], [0.43%],
    [Our LoRA full], [32], [77.41%], [0.83%],
    [Our DoRA full], [16], [*79.20%*], [*0.43%*],
  ),
) <tab:main-results>

The headline result directionally reproduces the paper's central finding. Full-scope rank-halved DoRA improved macro accuracy from 77.41% to 79.20%, a +1.79 point gain, while reducing trainable parameters from 0.83% to 0.43%. Our LoRA baseline is close to the paper's LoRA row (77.41% vs. 77.60%), suggesting that the evaluation pipeline is reasonably calibrated. Our DoRA score is below the paper's 80.50% reference, which is expected given the reduced data scale and limited search budget.

#figure(
  placement: none,
  caption: [Scope ablation],
  table(
    columns: (0.3fr, 0.2fr, 0.2fr, 0.2fr),
    align: (left, right, right, right),
    stroke: (_, y) => if y <= 1 { (top: 0.45pt) } else { none },
    table.header[Scope][LoRA][DoRA][Delta],
    [Full], [77.41%], [79.20%], [*+1.79*],
    [Attention], [79.59%], [78.09%], [-1.50],
    [MLP], [79.05%], [79.20%], [+0.15],
  ),
) <tab:ablations>

The ablation study is our main independent finding. DoRA is not universally better than LoRA under every adapter placement. Full-scope DoRA won or tied seven of eight tasks, with its largest gains on HellaSwag (+6.0 points), ARC-Challenge (+3.68), and OpenBookQA (+2.40). Attention-only DoRA lost on seven of eight tasks and fell by -1.50 macro points. MLP-only DoRA was roughly tied, gaining only +0.15 macro points. This changes the interpretation from "DoRA is better" to "DoRA helps most when the adapter scope covers enough transformer computation."

The rank sweep is hypothesis-generating rather than a final claim. Rank-8 full-scope DoRA reached 80.02% macro accuracy with only 0.221% trainable parameters, outperforming rank-16 in our one-seed runs. Without repeated seeds or a learning-rate grid, we do not claim that rank 8 is generally optimal. The broader lesson is that PEFT methods should be evaluated as full recipes including target modules, rank, learning rate, quantization, and evaluation extraction, not only as adapter formulas.

= Reflections

Reimplementing DoRA exposed details that wrapper-based experiments tend to hide. Correctness depended on target-module names, quantized weight dequantization, magnitude initialization, bias handling, prompt formatting, answer extraction, and adapter checkpoint compatibility. These engineering details were not peripheral. They determined whether the reproduction was meaningful.

The project also clarified the limits of our evidence. We did not reproduce the paper's full data scale, seed count, or training budget, so our absolute numbers should not be read as replacements for the official table. The controlled local comparison is still informative: LoRA lands close to the paper, full-scope DoRA improves in the expected direction with fewer trainable parameters, and the ablations identify adapter placement as a key condition for that improvement.

Given more time, we would train on the full Commonsense 170k dataset, repeat each condition across seeds, compare directly against the NVlabs implementation @nvlabsdora, and run a controlled rank/learning-rate grid. We would also add error analysis for HellaSwag and ARC-Challenge to determine whether the largest DoRA gains reflect better physical commonsense, answer calibration, or benchmark-specific artifacts.
