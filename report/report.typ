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
  heading-size: 12pt,
  abstract-size: 11pt,
  caption-size: 11pt,
  reference-size: 11pt,
  title-size: 17pt,
  author-name-size: 11pt,
  author-detail-size: 10pt,
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

Full fine-tuning adapts every parameter of a large language model (LLM), making each downstream task expensive to train and store. Parameter-efficient fine-tuning (PEFT) instead freezes the base model and trains small adapter modules. Our project reimplements a central result from the paper *DoRA: Weight-Decomposed Low-Rank Adaptation* by Liu et al. @liu2024dora, which extends LoRA: Low-Rank Adaptation of Large Language Models @hu2022lora by separating the direction and magnitude of an adapted weight. The LoRA baseline in the DoRA paper uses hyperparameters from a subsequent paper @llmadapters following the first LoRA paper.

LoRA keeps a pretrained weight $W$ frozen and learns a low-rank residual $Delta W = B A$. DoRA keeps the low-rank directional update but learns a per-output magnitude vector $m$:

$ W' = m dot (W + Delta W) / norm(W + Delta W). $

Our contribution is a reproduction of DoRA adjusted for the smaller training set and the Colab runtime with 4-bit NF4 quantization enabled to make the 7B experiments practical under our compute constraints. We implemented both LoRA and DoRA from scratch in PyTorch @paszke2019pytorch, reproduced the DoRA paper's main LoRA-vs-DoRA comparison on Meta's LLaMA2-7B model @touvron2023llama, and added scope ablations plus a supplementary rank sweep to better understand where DoRA helps. Our LoRA implementation uses the same hyperparameters as the DoRA paper's LoRA baseline instead of the first LoRA paper.

= Chosen Result

We targeted Table 1 of the DoRA paper for the LLaMA2-7B model, comparing LoRA rank 32 against DoRA rank 16 halved on all 8 benchmarks used in the paper. This table is central to the paper because it supports DoRA's main claim that it has higher accuracy than LoRA with fewer trainable parameters. The official result reports macro accuracy rising from 77.6% for LoRA to 80.5% for DoRA rank 16, while trainable parameters fall from 0.83% to 0.43%.

We chose it because it is quantitative, tied directly to the paper's main contribution, and feasible under our compute and time constraints. Our main scale change was training on a 15k commonsense subset rather than the full training set. The matched LoRA and DoRA runs still test the core comparison because they share the same model, data subset, prompts, evaluation code, and seed. We also added supplementary experiments beyond the main reproduction, including adapter-placement ablations and a rank sweep inspired by the paper's rank robustness study.

= Methodology

#figure(
  placement: none,
  caption: [Experimental setup],
  table(
    columns: (0.25fr, 0.70fr),
    align: (left, left),
    stroke: (_, y) => if y <= 1 { (top: 0.45pt) } else { none },
    table.header[Component][Choice],
    [Model], [Meta LLaMA2-7B],
    [Training data],
    [AGI-Edgerunners/LLM-Adapters Commonsense 15k fine-tuning file @agiedgerunnersdata],

    [Benchmarks],
    [BoolQ, PIQA, Social IQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA],

    [Runtime], [Google Colab A100],
    [Main comp.], [LoRA rank 32 vs. DoRA rank 16],

    [Training/eval],
    [3 epochs, cutoff length 256, seed 42, adapter-only checkpoints, per-task and macro accuracy],
  ),
) <tab:setup>

The implementation uses Hugging Face Transformers only for model and tokenizer loading @wolf2020transformers and Hugging Face Datasets for dataset materialization @lhoest2021datasets. The PEFT logic is implemented using PyTorch code, not importing PEFT wrapper code. Our LoRA adapter wraps a frozen base linear layer and adds a learned low-rank residual. Our DoRA adapter initializes a trainable per-output-channel magnitude from the pretrained weight norm, adds the low-rank update to the frozen weight, normalizes the resulting direction, and rescales it by the learned magnitude.

Beyond adapter math, the system implements in-place adapter injection after freezing the base model, adapter-only checkpoints, reloadable configs, standard linear and 4-bit quantized linear targets, benchmark evaluation, and result aggregation. For our additional experiment, the ablation study, we evaluated three adapter scopes including full adapters on query/key/value/up/down projections, attention-only adapters on query/key/value, and MLP-only adapters on up/down. This tests whether DoRA is a local formula improvement or depends on broader transformer coverage. The full scope matches the DoRA paper's LoRA and DoRA settings, while the original LoRA paper use then attention-only scope with addition of the output projection.

We controlled the comparison by keeping the base model, 15k subset, prompt, evaluation code, seed, benchmark files, and macro-average metric fixed across matched LoRA and DoRA runs. We used a Google Colab A100 GPU @bisong2019colab with 4-bit NF4 quantization @dettmers2023qlora to reduce the model's memory footprint. This quantized setup can introduce noise and slightly shift absolute scores, so it may keep us from reproducing the paper exactly, but it should not change the overall trend of the comparison. Additionally, correctness of the reimplementation is supported by passing a test suite with 50 tests covering adapter math, injection, checkpointing, config loading, data processing, evaluation, and CLI behavior. The main limitations are practical, including one seed, 15k rather than 170k training examples, and no exhaustive rank or learning-rate sweep.

= Results & Analysis

#figure(
  placement: none,
  caption: [Paper result versus our full-scope reproduction. Macro is the unweighted average accuracy of the eight commonsense tasks.],
  table(
    columns: (0.4fr, 0.15fr, 0.2fr, 0.2fr),
    align: (left, right, right, right),
    stroke: (_, y) => if y <= 1 { (top: 0.45pt) } else { none },
    table.header[Condition][Rank][Macro][Trainable],
    [Paper LoRA], [32], [77.60%], [0.83%],
    [Paper DoRA], [16], [80.50%], [0.43%],
    [Our LoRA full], [32], [77.41%], [0.83%],
    [Our DoRA full], [16], [*79.20%*], [*0.43%*],
  ),
) <tab:main-results>

Full-scope rank-halved DoRA improved macro accuracy from 77.41% to 79.20% while reducing trainable parameters from 0.83% to 0.43%, directionally reproducing the paper's central finding. Our LoRA baseline is close to the official LoRA reference (77.41% vs. 77.60%), suggesting that the evaluation pipeline is aligned. Our DoRA score is lower than the 80.50% reference, which is plausible given the smaller training set, one seed, NF4 runtime, and limited hyperparameter search.

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

The ablation study is our main independent experiment. DoRA is not universally better than LoRA across all adapter placements. Full-scope DoRA won or tied seven of eight tasks, with its largest gains on HellaSwag (+6.0 points), ARC-Challenge (+3.68), and OpenBookQA (+2.40). Meanwhile, attention-only DoRA lost on seven of eight tasks and fell by -1.50 macro points. MLP-only DoRA was roughly tied, gaining only +0.15 macro points. This shows that DoRA's improvements over LoRA depend on adapter placement and that DoRA tends to perform better when applied broadly across the transformer.

The rank sweep is supplementary evidence rather than the main reproduced claim. The DoRA paper itself studies rank robustness in section 5.5 by varying the rank size (4, 8, 16, 32, 64), and our sweep follows that idea, but scaled down to 4, 8, and 16.

#figure(
  placement: none,
  image("../results/fig6_rank_scaling_report.png", width: 100%),
  caption: [Full-scope rank sweep. DoRA keeps higher macro accuracy than LoRA at matched ranks 4, 8, and 16 in our one-seed reduced-data setting.],
) <fig:rank-scaling>

Rank 8 full scope DoRA reached the highest macro accuracy in our sweep while using only 0.221% trainable parameters. Without repeated seeds or a learning-rate grid, we do not claim that rank 8 is generally optimal. The more safe takeaway from @fig:rank-scaling is that DoRA remains above LoRA at every matched rank, and LoRA's accuracy falls faster as rank decreases. This follows the same general trend as the official DoRA paper's rank robustness study, while using our smaller LLaMA-2-7B reproduction setting.

= Reflections

Reimplementing DoRA taught us that the difficulty was not only in the formula but also in assuring that the entire experimental pipeline was sufficiently faithful to the comparison to make it meaningful. Instead of relying on PEFT wrappers, we had to implement LoRA and DoRA adapter math, module injection, quantized linear-layer support, adapter-only checkpointing, prompt formatting, answer extraction, evaluation, and result aggregation. Small choices, such as target-module names, magnitude initialization, dequantizing 4-bit weights before normalization, and bias handling, could have changed the results, so unit tests and controlled configs became necessary parts of the research process.

The project improved through several iterations. We first focused on adapter correctness and reloadable checkpoints, then reproduced the main controlled LoRA-vs-DoRA comparison, then added scope ablations, and finally ran a smaller rank sweep. The main takeaway is that our full-scope DoRA result supports the paper's central claim that DoRA improved over full-scope LoRA while using fewer trainable parameters. At the same time, the ablation experiments made the story more detailed. Attention-only LoRA scored higher than our full-scope DoRA in this one-seed, reduced-training-data setting, but it still scored lower than the official DoRA paper's 80.50% result. Because our setup used 15k examples, 4-bit quantization, a single seed, and no exhaustive hyperparameter sweep, this should be treated as an interesting lead on adapter placement, not as evidence that attention-only LoRA is generally better than DoRA.

Given more time, we would train on the full Commonsense 170k dataset, repeat every condition across multiple seeds, compare directly against the NVlabs implementation @nvlabsdora, and run a rank/learning-rate grid for both methods. We would also add error analysis for HellaSwag and ARC-Challenge to test whether DoRA's largest gains reflect better commonsense reasoning, calibration differences, or benchmark artifacts. The larger lesson is that PEFT methods should be evaluated on average accuracy, adapter placement, rank needs, and stability, while accounting for realistic compute limits.
