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
  bibliography: bibliography("refs.bib"),
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

Full fine-tuning adapts every parameter of a large language model (LLM), making each downstream task expensive to train and store. Parameter-efficient fine-tuning (PEFT) instead freezes the base model and trains small adapter modules. Our project reimplements a central result from the paper *DoRA: Weight-Decomposed Low-Rank Adaptation* by Liu et al. @liu2024dora, which extends LoRA: Low-Rank Adaptation of Large Language Models @hu2022lora by separating the direction and magnitude of an adapted weight. The LoRA baseline in the DoRA paper uses hyperparameters from subsequent paper @llmadapters following the original LoRA paper.

LoRA keeps a pretrained weight $W$ frozen and learns a low-rank residual $Delta W = B A$. DoRA keeps the low-rank directional update but learns a per-output magnitude vector $m$:

$ W' = m dot (W + Delta W) / norm(W + Delta W). $

Our contribution is a reproduction of DoRA adjusted for the smaller training set and the Colab runtime with 4-bit NF4 quantization enabled to make the 7B experiments practical under our compute constraints. We implemented both LoRA and DoRA from scratch in PyTorch, reproduced the DoRA paper's main LoRA-vs-DoRA comparison on Meta's LLaMA-2-7B model, and added scope ablations plus a supplementary rank sweep to better understand where DoRA helps. Our LoRA implementation uses the same hyperparameters as the DoRA paper's LoRA baseline instead of the original LoRA paper.

= Chosen Result

We targeted Table 1 of the DoRA paper for the LLaMA-2-7B model, comparing LoRA rank 32 against rank-halved DoRA rank 16 on all 8 benchmarks that the paper used. This table is central to the paper because it supports DoRA's main claim that it has higher accuracy than LoRA with fewer trainable parameters. The official result reports macro accuracy rising from 77.6% for LoRA to 80.5% for DoRA rank 16 while trainable parameters fall from 0.83% to 0.43%.

We chose this result because it is quantitative, directly tied to the paper's main contribution, and feasible to investigate under our compute constraints and time. Our deliberate scale change was training on a 15k commonsense subset instead of the paper's 170k training set. This smaller setting still tests the core claim because matched LoRA and DoRA runs share the same base model, data subset, prompts, benchmark files, seed, and evaluation. We also added supplementary experiments beyond the main reproduction, including adapter-placement ablations and a rank sweep that aligns with section 5.5 in the DoRA paper.

= Methodology

#figure(
  placement: none,
  caption: [Experimental setup],
  table(
    columns: (0.25fr, 0.70fr),
    align: (left, left),
    stroke: (_, y) => if y <= 1 { (top: 0.45pt) } else { none },
    table.header[Component][Choice],
    [Model], [Meta LLaMA-2-7B, frozen base weights],
    [Training data], [Commonsense 15k instead of the paper's 170k version],

    [Benchmarks],
    [BoolQ, PIQA, Social IQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA],

    [Runtime], [Google Colab A100 with 4-bit NF4 quantization],
    [Main comp.], [LoRA rank 32 vs. DoRA rank 16],

    [Training/eval],
    [3 epochs, cutoff length 256, seed 42, adapter-only checkpoints, per-task and macro accuracy],
  ),
) <tab:setup>

The implementation uses Hugging Face Transformers only for model and tokenizer loading @transformers. The PEFT logic is implemented form scratch in our repository using PyTorch. Our LoRA adapter wraps a frozen base linear layer and adds a learned low-rank residual. Our DoRA adapter initializes a trainable per-output-channel magnitude from the pretrained weight norm, adds the low-rank update to the frozen weight, normalizes the resulting direction, and rescales it by the learned magnitude.

The system also implements in-place adapter injection after freezing the base model, adapter-only checkpoints, reloadable configs, standard linear and 4-bit quantized linear targets, benchmark evaluation, and result aggregation. For our additional experiment, the ablation study, we evaluated three adapter scopes. Full adapters on query, key, value, up, and down projections. Then  Attention-only adapters on query, key, and value. Then finally on, MLP-only adapters on up and down projections. This ablation tests whether DoRA is a local formula improvement or whether it needs broad transformer coverage. The full scope matches the DoRA paper's settings for both LoRA and DoRA, but it is important to note the original LoRA paper uses the attention-only scope with the addition of the the output projection.

We kept the comparison controlled by keeping the base model, 15k subset, prompt and evaluation code, seed, benchmark files, and macro average metric fixed across matched LoRA and DoRA runs. We used a Google Colab A100 GPU with 4-bit NF4 quantization to reduce the model's memory footprint. This quantized setup can add some noise and shift absolute scores slightly, so it may keep us from matching the paper exactly, but it should not change the overall trend of the comparison. Additionally, correctness of the reimplementation is supported by passing a test suite with 50 tests covering adapter math, injection, checkpointing, config loading, data processing, evaluation, and CLI behavior. The main limitations are practical, including one seed, 15k rather than 170k training examples, and no exhaustive rank or learning-rate sweep.

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

The headline result directionally reproduces the paper's central finding. Full-scope rank-halved DoRA improved macro accuracy from 77.41% to 79.20%, a +1.79 point gain, while reducing trainable parameters from 0.83% to 0.43%. Our LoRA baseline is close to the DoRA paper's LoRA results (77.41% vs. 77.60%), suggesting that the evaluation pipeline is accurate. Our DoRA score is below the paper's 80.50% reference, which is expected given the reduced training data scale, the small amount of noise introduced by the quantized runtime, and the fact that we did not run an exhaustive hyperparameter sweep.

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

The ablation study is our main independent experiment. DoRA is not universally better than LoRA under every adapter placement. Full-scope DoRA won or tied seven of eight tasks, with its largest gains on HellaSwag (+6.0 points), ARC-Challenge (+3.68), and OpenBookQA (+2.40). Meanwhile, attention-only DoRA lost on seven of eight tasks and fell by -1.50 macro points. MLP-only DoRA was roughly tied, gaining only +0.15 macro points. This shows that DoRA's improvements over LoRA are dependent on the adapter placement and DoRA tends to do better when applied broadly across the transformer.

The rank sweep is supplementary evidence rather than the main reproduced claim. The DoRA paper itself studies rank robustness in section 5.5 by varying the rank size (over 4, 8, 16, 32, 64), and our sweep follows that idea but slightly scaled down testing 4, 8, and 16.

Rank-8 full-scope DoRA reached 80.02% macro accuracy with only 0.221% trainable parameters, outperforming rank-16 in our one-seed runs. However, without repeated seeds or a learning-rate grid, we do not claim that rank 8 is generally optimal. Also, there is an expected amount of noise in the results. Even the official DoRA paper shows DoRA rank 8 performing slightly higher than DoRA rank 8, but this is on the LLaMA-7B model not the LLaMA-2-7B model which we have been using.

The bigger takeaway about the rank sweep is that DoRA's consistently has higher accuracy than LoRA at the same rank size. Also, that LoRA's accuracy drops off faster than DoRA's as the rank size decreases. Overall, the rank sweep results show the same general trend as the official DoRA paper's section 5.5, showing that DoRA is more robust to rank size reductions than LoRA.

= Reflections

Reimplementing DoRA taught us that the difficulty was not only in the formula, but in making the whole experimental pipeline faithful enough for the comparison to mean something. Instead of relying on PEFT wrappers, we had to implement LoRA and DoRA adapter math, module injection, quantized linear-layer support, adapter-only checkpointing, prompt formatting, answer extraction, evaluation, and result aggregation. Small choices such as target-module names, magnitude initialization, dequantizing 4-bit weights before normalization, and bias handling could have changed the result, so unit tests and controlled configs became a necessary part of the research process.

The project improved through several iterations. We first focused on adapter correctness and reloadable checkpoints, then reproduced the main controlled LoRA-vs-DoRA comparison, then added scope ablations, and finally ran a smaller rank sweep. The main takeaway is that our full-scope DoRA result supports the paper's central claim that DoRA improved over full-scope LoRA while using fewer trainable parameters. At the same time, the ablations made the story more nuanced. Attention-only LoRA scored higher than our full-scope DoRA in this one-seed, reduced training data setting, but it still scored lower than the official DoRA paper's 80.50% result. Because our setup used 15k examples, 4-bit quantization, one seed, and no exhaustive hyperparameter sweep, this should be treated as an interesting lead about adapter placement, not evidence that attention-only LoRA is generally better than DoRA.

Given more time, we would train on the full Commonsense 170k dataset, repeat every condition across multiple seeds, compare directly against the NVlabs implementation @nvlabsdora, and run a rank/learning-rate grid for both methods. We would also add error analysis for HellaSwag and ARC-Challenge to test whether DoRA's largest gains reflect better commonsense reasoning, calibration differences, or benchmark artifacts. The broader lesson is that PEFT methods should be evaluated by average accuracy, adapter placement, rank needs, and stability under realistic compute limits.
