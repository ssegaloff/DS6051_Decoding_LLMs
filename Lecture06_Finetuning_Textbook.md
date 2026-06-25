# Fine-Tuning LLMs — A First-Principles Walkthrough of Lecture 6

*DS 6051, Decoding Large Language Models (UVA, Summer 2026). A textbook-style
expansion of the Lecture 6 slide deck. The spine is the single tradeoff that
organizes the whole PEFT family: **model quality versus compute cost**. Each
technique earns its place by attacking whichever cost term the previous one
left dominant.*

---

## 0. Linear Algebra Refresh — Rank

This section is not on the slides, but nothing in LoRA, Q-LoRA, or DoRA makes
sense without it. Rank is the hinge the entire low-rank story turns on, so we
build it from the ground up.

### Rank is about independent directions, not shape

The *shape* of a matrix $W \in \mathbb{R}^{d \times k}$ tells you how many rows
and columns it has. Its **rank** tells you something completely different: how
many genuinely independent directions the matrix can produce. Formally, the
rank is the number of linearly independent columns — equivalently (and this
equivalence is a theorem, not a definition) the number of linearly independent
rows. Row rank always equals column rank.

The geometric reading is the one to keep in your head. Think of $W$ as a linear
map $x \mapsto Wx$ taking input vectors in $\mathbb{R}^k$ to output vectors in
$\mathbb{R}^d$. The set of *all* outputs the map can ever produce,
$\{Wx : x \in \mathbb{R}^k\}$, is called the **column space** or **image** of
$W$. Every output $Wx$ is a linear combination of $W$'s columns (the entries of
$x$ are just the mixing coefficients), so the image is exactly the span of the
columns. The **rank is the dimension of that image** — the number of
independent directions the output can point in.

A useful piece of intuition: imagine the matrix as a machine that takes a
high-dimensional input and reports it back through a limited number of dials. If
there are only two dials, then no matter how many input knobs you turn, the
output can only move within a two-dimensional plane. *Precisely:* if
$\operatorname{rank}(W) = r$, then the image of $W$ is an $r$-dimensional
subspace, and every output the matrix can ever produce lives in that subspace,
regardless of the input.

### Why a matrix can have high dimension but low rank

Shape and rank are decoupled. A matrix can be enormous and still have tiny rank,
because rank is capped by redundancy among the columns (or rows), not by their
count. The hard bound is

$$\operatorname{rank}(W) \le \min(d, k),$$

but the *actual* rank can be far below even that ceiling when the columns repeat
information.

Take

$$
W = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 3 & 6 & 9 \end{bmatrix}.
$$

This is a $3 \times 3$ matrix — full "dimension" in the sense of shape — yet its
rank is **1**. Every column is a scalar multiple of $(1,2,3)^\top$: the second
column is twice the first, the third is three times the first. There is really
only one independent direction here, dressed up in a $3\times3$ costume. The
image of this map is a single *line* through the origin in $\mathbb{R}^3$, not a
plane and not the whole space. A 65-billion-parameter weight matrix can, in
exactly this way, carry far less independent information than its parameter
count suggests — and that is the empirical observation LoRA exploits.

### The column-dependence example

Rank deficiency is just linear dependence among columns. Consider

$$
W = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix}.
$$

The first two columns, $(1,0,1)^\top$ and $(0,1,1)^\top$, are independent — neither
is a multiple of the other. But the third column is their sum:
$(1,0,1)^\top + (0,1,1)^\top = (1,1,2)^\top$. The third column therefore adds no
new direction. Any output $Wx$ can already be reached using only the first two
columns, so the image is a two-dimensional plane and
$\operatorname{rank}(W) = 2$. The third column is "free-riding" on the first
two. Rank counts the columns that actually pull their weight.

### Rank, image, and what the matrix can express

Tie these threads together: the rank is simultaneously (i) the number of
independent columns, (ii) the number of independent rows, and (iii) the
dimension of the image. The third framing is the one that matters downstream. A
rank-$r$ matrix is a map whose reachable outputs fill an $r$-dimensional
subspace and no more. If you want a weight matrix to be able to *express* a
change that points in many independent directions, you need a high-rank matrix.
If the change you need only points in a few directions, a low-rank matrix
suffices — and a low-rank matrix is cheap to store and cheap to train.

### Why $\operatorname{rank}(PQ) \le \min(\operatorname{rank}(P),\operatorname{rank}(Q))$

This inequality is the load-bearing fact for LoRA, so we prove it both ways.
We use generic names $P$ and $Q$ here to avoid any conflict with the $B$ (tall)
and $A$ (wide) convention LoRA introduces in Section 5.

Let $P \in \mathbb{R}^{d \times r}$ and $Q \in \mathbb{R}^{r \times k}$, so the
product $PQ$ is $d \times k$. Write the product as the composed map
$x \mapsto P(Qx)$.

**Bound by $\operatorname{rank}(P)$ (the columns argument).** Each column of $PQ$
is $P$ applied to the corresponding column of $Q$ — that is, each column of $PQ$
is a linear combination of the columns of $P$ with coefficients drawn from a
column of $Q$. So every column of $PQ$ lives in the column space of $P$, which
gives $\operatorname{rank}(PQ) \le \operatorname{rank}(P)$.

**Bound by the intermediate dimension (the bottleneck argument).** Reading the
composition left to right: $Q$ first maps the input into $\mathbb{R}^r$, and
then $P$ maps that into $\mathbb{R}^d$. Whatever $P$ does afterward, its input
already lives in a space of dimension at most $r$, and a linear map cannot
*increase* the dimension of a subspace — it can only preserve or collapse it.
So the image of $PQ$ has dimension at most $r$, hence at most
$\operatorname{rank}(Q)$ (which is itself $\le r$). Combining,
$\operatorname{rank}(PQ) \le \min(\operatorname{rank}(P), \operatorname{rank}(Q)) \le r$.

The phrase to remember is **information bottleneck**. No matter how expressive
$P$ and $Q$ are individually, routing the signal through an $r$-dimensional
waist between them caps the rank of their product at $r$. You cannot reconstruct
more independent directions on the far side than survived the squeeze in the
middle.

### Why this makes the LoRA bottleneck work

LoRA represents a weight *update* as a product $\Delta W = BA$ with
$B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, where $r$ is
small (often 4, 8, 16) and $\min(d,k)$ is large (often thousands). Identifying
$B$ with the left factor $P$ and $A$ with the right factor $Q$ from the proof
above, $\operatorname{rank}(\Delta W) = \operatorname{rank}(BA) \le r$. The update is
*structurally incapable* of being anything but low rank — it is forced through
the $r$-dimensional waist. The parameter count drops from $dk$ (a full update)
to $r(d+k)$ (the two factors), and the whole bet is that the update a model
needs in order to specialize to a task has **low intrinsic rank** — that
adaptation moves the weights along only a handful of independent directions.
Section 5 makes that bet precise. For now, the only thing you need is the cap:
a product through an $r$-dimensional bottleneck has rank at most $r$.

---

## 1. Why Fine-Tuning Exists and What Problem It Solves

### The problem

Pretraining produces a model that is a brilliant generalist and a poor
specialist. A base LLM has absorbed broad linguistic competence and world
knowledge from a vast unlabeled corpus by doing one thing — predicting the next
token — but it has not been shaped toward any particular task, domain
vocabulary, output format, safety posture, or house style. Pretraining is also
ruinously expensive and data-hungry: it is not something you redo whenever you
need the model to behave a little differently.

Fine-tuning is the answer to "how do I get task-specific competence without
paying the pretraining bill again?" It is transfer learning applied to LLMs:
take the pretrained weights as a starting point and continue training on a
smaller, targeted dataset so the model specializes. *Precisely:* fine-tuning
re-optimizes some or all of the model's parameters on a downstream objective,
inheriting the representations learned during pretraining instead of learning
them from scratch. The representations transfer; only the specialization is new.

### Fine-tuning's place in the LLM pipeline

The deck frames fine-tuning as one stage of a larger pipeline (Parthasarathy et
al., 2024): data is collected and prepared, the model is pretrained,
**fine-tuned**, aligned, evaluated, and finally deployed and monitored.
Fine-tuning sits between the general-purpose base and the deployed,
task-specific system. Everything in this lecture concerns that one stage.

### The seven-stage fine-tuning pipeline

The deck's seven-stage view expands fine-tuning itself into a workflow: (1)
dataset preparation, (2) model initialization from pretrained weights, (3)
training-environment setup, (4) the fine-tuning run itself, (5) evaluation and
validation, (6) deployment, and (7) monitoring and maintenance. The pedagogical
point is that the gradient-descent step everyone fixates on is a single stage in
a loop dominated by data work and evaluation — the parts that actually decide
whether fine-tuning helps.

### Best practices and pitfalls

The deck lists four best practices — **data quality and quantity**,
**hyperparameter tuning**, **regular evaluation**, and attention to
**computational requirements** — and three pitfalls. The pitfalls deserve
first-principles treatment because they recur throughout the lecture:

- **Overfitting.** With a small fine-tuning set and a high-capacity model, the
  optimizer can memorize the training examples rather than learn the task. The
  model's loss on training data keeps falling while held-out performance
  stalls or degrades. The cure is the usual one — regularization, early
  stopping, more or better data — but the risk is sharper in fine-tuning
  precisely because the datasets are small.
- **Underfitting.** The opposite failure: too few training steps, too low a
  learning rate, or too little trainable capacity (a relevant knob once we
  reach PEFT, where you deliberately restrict capacity) leaves the model
  unable to capture the task.
- **Catastrophic forgetting.** When you update *all* the weights on a narrow
  task, you overwrite the very parameters that encoded the model's general
  competence. The model gets better at the new task and measurably worse at
  everything else. This is the single most important pitfall for motivating
  PEFT: **if you never touch the pretrained weights, you cannot forget what
  they encode.** Hold onto this — it reappears as a structural guarantee in
  every PEFT method.

### What you actually fine-tune (the knobs)

The deck enumerates the controllable surface: optimizers, learning rate and
its schedule, memory-optimization techniques (**gradient checkpointing**,
which trades recomputation for activation memory; **gradient accumulation**,
which simulates a large batch on small hardware; and **quantization**, which we
develop fully in Section 6), regularization, batch size, **noise embeddings**
(adding noise to input embeddings during training, e.g. NEFTune, to improve
robustness), and **label smoothing** (softening one-hot targets to reduce
overconfidence). Several of these — checkpointing, accumulation, quantization —
exist purely to fit training into a fixed memory budget, which is the cost side
of the tradeoff this entire lecture circles.

### Types of fine-tuning

The deck names five families: **supervised fine-tuning (SFT)**, **prefix
tuning**, **parameter-efficient fine-tuning (PEFT)**, **continual learning**,
and **knowledge distillation**. SFT and PEFT (with its sub-methods) are the
focus; the others are surveyed at the end.

**Supervised fine-tuning** is the workhorse. You assemble a labeled dataset of
(input, desired-output) pairs and train the model to produce the correct output
for each input — instruction tuning is SFT where the "labels" are good
responses to instructions. It adapts a general model to a specific task by
direct example. Everything else in the lecture is, in effect, a way to do SFT
*more cheaply* by changing which parameters move and how they are stored.

**Prefix tuning** (Li and Liang, 2021) is the first method we meet that freezes
the transformer entirely. Instead of editing weights, it *prepends a short
sequence of trainable continuous vectors* — "virtual tokens," not real words —
to the input, and (in the full formulation) to the keys and values at every
layer. The frozen model attends to these learned prefix vectors as if they were
context, and the prefix steers its behavior. Geometrically, you are not moving
the model's parameters at all; you are searching the *input/activation space*
for a context that elicits the task you want.

The payoff is modularity and low-data robustness. One frozen base serves many
tasks: swap in a different prefix per task (translation, summarization,
table-to-text) instead of storing a full fine-tuned copy each time. The deck's
results show prefix tuning matching or beating full fine-tuning on
ROUGE-1/2, BLEU, and ROUGE — and the gap *widens in favor of prefix tuning as
the training set shrinks*. When data is scarce, freezing the base and learning
only a tiny prefix overfits less than re-estimating hundreds of millions of
weights from a handful of examples. This is the same logic — freeze the base,
train something small, avoid forgetting and overfitting — that the rest of PEFT
formalizes.

---

## 2. The Cost of Fine-Tuning — Where Every Bit Goes

This is the section that makes the rest of the lecture inevitable. We work the
memory arithmetic for full fine-tuning, LoRA, and Q-LoRA, and watch the
dominant cost term move each time.

### Full fine-tuning: 96 bits per parameter

Full fine-tuning updates every parameter, so for **each** parameter the GPU must
hold several quantities resident in memory during a training step. With 16-bit
weights and the Adam optimizer (the deck's example):

- **Weight:** 16 bits (2 bytes). The parameter itself.
- **Weight gradient:** 16 bits (2 bytes). One gradient per parameter, same
  precision as the weight.
- **Optimizer states:** 64 bits (8 bytes). Adam maintains two running
  statistics per parameter — the first moment (mean of gradients) and the
  second moment (mean of squared gradients) — and keeps them in 32-bit
  precision for numerical stability. Two values × 4 bytes = 8 bytes = 64 bits.
  (The deck's "65 bits" is a typo for 64. Note also that true mixed-precision
  setups often keep an additional fp32 *master copy* of the weights, pushing
  the real figure higher; the deck uses the leaner 8-byte accounting.)

Adding these: $16 + 16 + 64 = 96$ bits $= 12$ bytes **per parameter**.

Now scale it. A 65-billion-parameter model needs

$$
65 \times 10^{9} \text{ params} \times 12 \text{ bytes} = 780 \times 10^{9}
\text{ bytes} = 780 \text{ GB}.
$$

At roughly 45 GB usable per data-center GPU, that is about **17 data-center
GPUs** (or ~34 consumer GPUs at ~24 GB each). And notice *where the cost lives*:
of the 96 bits, only 16 are the weights themselves. The other **80 bits — the
gradient and the optimizer states — exist solely because you chose to train
every parameter.** That 80-bit overhead is the target.

### LoRA: 17.6 bits per parameter

LoRA (Section 5) freezes the base weights and trains only small low-rank
adapters. The accounting per base-model parameter becomes:

- **Weight:** 16 bits. The frozen base weights still have to be resident for the
  forward pass — you compute with them even though you never update them. This
  term does not shrink.
- **Weight gradient:** ~0.4 bits.
- **Optimizer state:** ~0.8 bits.
- **Adapter weights:** ~0.4 bits.

Total: ~**17.6 bits per parameter**. The three small terms are approximate and
their exact values depend on the adapter fraction and optimizer, but the
*structure* is the whole point. Gradients, optimizer states, and the adapter
parameters only exist for the **trainable** parameters, and LoRA trains on the
order of 1% of them. Amortized across all base parameters, those terms collapse
from full fine-tuning's 80 bits to about **1.6 bits** — a roughly fiftyfold
reduction in training overhead — because you froze the base and stopped paying
gradient-and-optimizer rent on 99% of the weights.

For the 65B model:

$$
65 \times 10^{9} \times 17.6 \text{ bits} = 65 \times 10^{9} \times 2.2
\text{ bytes} = 143 \text{ GB},
$$

about **4 data-center GPUs** (or ~8 consumer GPUs). LoRA crushed the optimizer
term. But look at what is now dominant: of 17.6 bits, **16 are the frozen base
weights**. LoRA solved the overhead problem and handed us a new bottleneck — the
sheer cost of *storing* the base weights in 16-bit. That is the next target.

### Q-LoRA: 5.2 bits per parameter

Q-LoRA (Section 6) keeps LoRA's freezing trick and adds one move: store the
frozen base weights in **4-bit** precision instead of 16-bit.

- **Weight:** 4 bits. The frozen base, quantized 4×.
- **Weight gradient:** ~0.4 bits.
- **Optimizer state:** ~0.8 bits.
- **Adapter weights:** ~0.4 bits.

Total: ~**5.2 bits per parameter** (as the slide states). Note that the four
listed components — 4 + 0.4 + 0.8 + 0.4 — sum to ~5.6, not 5.2; the slide
carries the same inconsistency. The 42 GB figure is consistent with 5.2 bits,
so the total and the GB footprint are internally consistent with each other; the
per-component breakdown is approximate and the ~0.4/~0.8 figures are themselves
rounded, so the small discrepancy is an artifact of rounding. The structural
point stands: the trainable overhead (~1.6 bits, unchanged from LoRA) is now
paired with a 4-bit base rather than a 16-bit one.

For the 65B model:

$$
65 \times 10^{9} \times 5.2 \text{ bits} = 65 \times 10^{9} \times 0.65
\text{ bytes} \approx 42 \text{ GB},
$$

which fits on a **single** GPU. The deck's annotation literally crosses out
"780 GB → 17×" and replaces it with "42 GB → 1×": fine-tuning a 65B model went
from a 17-GPU data-center job to a single-GPU job.

### The whole story in one table

| | Weight | Gradient | Optimizer | Adapter | **Total/param** | **65B footprint** | GPUs |
|---|---|---|---|---|---|---|---|
| **Full FT (16-bit)** | 16 | 16 | 64 | — | **96 bits** | 780 GB | ~17 |
| **LoRA** | 16 | ~0.4 | ~0.8 | ~0.4 | **17.6 bits** | 143 GB | ~4 |
| **Q-LoRA** | 4 | ~0.4 | ~0.8 | ~0.4 | **5.2 bits** | 42 GB | 1 |

Read the table column by column and the throughline is unmistakable. Full
fine-tuning's cost is dominated by the **gradient + optimizer** terms on every
parameter. **LoRA attacks those** by freezing the base, collapsing 80 bits of
overhead to ~1.6. That leaves the **frozen base weight** as the dominant term,
so **Q-LoRA attacks that** by quantizing 16 bits down to 4. Each technique
targets exactly the cost the previous one left standing. Everything that
follows is a variation on this theme: spend trainable capacity and bits only
where they buy quality, and freeze or compress everywhere else.

---

## 3. Parameter-Efficient Fine-Tuning as a Family

Parameter-efficient fine-tuning (PEFT) is not a single method but a design
principle with several incarnations. The principle: **freeze the vast majority
of the pretrained weights and introduce a small number of new trainable
parameters**, so that the gradient, optimizer, and storage costs scale with the
small trainable set rather than the enormous frozen one.

The deck's PEFT roster is the agenda for the rest of the lecture:

- **Adapters** — insert small trainable modules between frozen sublayers.
- **LoRA** — reparameterize the weight *update* as low-rank.
- **Q-LoRA** — LoRA on top of a 4-bit-quantized frozen base.
- **DoRA** — split each weight into magnitude and direction, and LoRA-adapt the
  direction.
- **Multiple adapters** — one frozen base serving many swappable task modules.

Every member shares three consequences that follow directly from the
freeze-and-add principle. First, **drastically reduced training memory**, for
exactly the reason the Section 2 arithmetic showed — gradients and optimizer
state are needed only for the few trainable parameters. Second, **structural
resistance to catastrophic forgetting**: the pretrained weights are never
overwritten, so the model's general competence is preserved by construction
rather than by careful tuning. Third, **portable, composable task modules**:
because the trainable part is tiny (often a few megabytes) and the base is
shared, you can store and swap many specializations, which Section 8 turns into
a deployment strategy. PEFT is the realization of "freeze the base, train
something small" that prefix tuning previewed, generalized into a family.

---

## 4. Adapters

### Why they exist

Adapters (Houlsby et al., 2019) are the historical root of the PEFT family and
the cleanest illustration of the freeze-and-add principle. The problem they
address is full fine-tuning's twin costs: you pay the gradient/optimizer
overhead on every parameter, *and* you end up with an entire fine-tuned copy of
the model per task. Adapters keep one frozen base and attach small trainable
modules, so the only per-task cost is the adapters.

### How they work

The deck's diagram (Houlsby et al.) shows the architecture. Inside each
transformer layer, **two** adapter modules are inserted. Reading bottom-up: the
attention sublayer feeds into a residual add, and the adapter sits immediately
*after* that add but *before* the subsequent Layer Norm. The same pattern repeats
after the feed-forward sublayer: feed-forward → residual add → **Adapter** →
Layer Norm. Every original transformer weight is frozen; only the adapters
train.

Each adapter is a **bottleneck**: it takes the $d$-dimensional hidden vector,
**down-projects** it to a much smaller dimension $m \ll d$, applies a
**nonlinearity**, **up-projects** back to $d$, and adds the result to its input
through a **skip connection**. The bottleneck is what keeps the parameter count
low — the two projections together hold about $2dm$ parameters, versus the
$d^2$ of a full layer, and with $m$ a small fraction of $d$ this is a tiny
addition.

The skip connection is doing quiet but important work. Because the adapter's
output is *added* to its input, and the adapter's weights are initialized small
(near zero), the module starts as an approximate **identity** — at the first
step it barely perturbs the frozen network. Training then nudges it away from
identity only as far as the task requires. This near-identity initialization is
the recurring PEFT trick for stability: begin at the pretrained function, and
move only as needed. (LoRA's $B=0$ initialization in Section 5 is the same idea
in a different guise.)

### Cost and the weakness that motivates LoRA

Adapters typically make well under 10% of parameters trainable while reaching
near-full-fine-tuning quality, and they give you the one-frozen-base,
small-per-task-module structure. But they have a specific defect, and naming it
sets up the next method exactly.

Adapters are inserted **into the forward pass as extra sequential layers**. They
add depth to the computation graph — a down-projection, a nonlinearity, and an
up-projection that every token must pass through, in series, at every layer.
That depth cannot be folded away: it shows up as **added inference latency**,
and because the adapter is a genuine extra nonlinear stage, you cannot merge it
back into the original weight matrices to recover the base architecture. You pay
for the adaptation on every forward pass, forever. LoRA was designed to keep
adapters' parameter efficiency while *eliminating this latency*.

---

## 5. LoRA — Low-Rank Adaptation

### Why it exists

LoRA (Hu et al., 2021) keeps the freeze-and-add philosophy but changes *what*
you add and *where*. Instead of inserting a new nonlinear module into the
forward path — adapters' source of latency — LoRA reparameterizes the **weight
update itself** as a low-rank matrix that sits *alongside* a chosen weight, in
parallel rather than in series. Because the update is purely linear and
parallel, it can be merged into the original weight after training, so inference
runs on the original architecture at the original speed. LoRA is, in a sense,
adapters with the nonlinearity removed and the placement changed so the cost
vanishes at inference.

### The decomposition $W_0 + BA$

Take any weight matrix $W_0 \in \mathbb{R}^{d \times k}$ you want to adapt. Full
fine-tuning would replace it with $W_0 + \Delta W$, where $\Delta W$ is an
unconstrained $d \times k$ update with $dk$ free parameters. LoRA constrains the
update to be low-rank by writing it as a product:

$$
\Delta W = BA, \qquad B \in \mathbb{R}^{d \times r}, \quad A \in \mathbb{R}^{r
\times k}, \quad r \ll \min(d,k).
$$

The adapted forward pass is

$$
h = W_0 x + \Delta W x = W_0 x + B(Ax).
$$

The base weights $W_0$ stay frozen; only $A$ and $B$ are trained. The deck's
diagram contrasts this with regular fine-tuning: where regular fine-tuning adds
a full-rank "weight update $\Delta W$" block, LoRA adds the two trapezoids $B$
and $A$ meeting at the narrow waist of width $r$.

### Rank as the bottleneck (the formal core)

This is where Section 0 pays off directly. By the rank inequality,

$$
\operatorname{rank}(\Delta W) = \operatorname{rank}(BA) \le \min(\operatorname{rank}(B),
\operatorname{rank}(A)) \le r.
$$

The update is *structurally* rank-$\le r$. Read $B(Ax)$ as a two-stage map: $A$
projects the $k$-dimensional input down into an $r$-dimensional subspace (its
row space), and $B$ lifts that $r$-dimensional signal back out into the
$d$-dimensional output space (its column space). The adaptation can therefore
read only $r$ independent linear combinations of the input and can write only
into an $r$-dimensional subspace of the output. That is the geometric meaning of
the bottleneck: the update lives entirely inside an $r$-dimensional corridor
through weight space.

The hypothesis that justifies this — building on Aghajanyan et al.'s observation
that fine-tuning has low *intrinsic dimensionality* — is that the weight change
needed to specialize a pretrained model genuinely has low rank. Adaptation moves
the model along only a few independent directions, so a rank-$r$ update with
small $r$ recovers most of what full fine-tuning would do. LoRA is a bet that the
intrinsic rank of $\Delta W$ is low, and the bet pays off empirically.

The parameter savings are immediate. The full update costs $dk$ parameters; the
factored update costs $r(d+k)$. With $d = k = 4096$ and $r = 8$:

$$
dk = 4096^2 = 16{,}777{,}216, \qquad r(d+k) = 8 \times 8192 = 65{,}536,
$$

a **256-fold** reduction in trainable parameters for that one matrix — and the
gradient and optimizer memory shrink in proportion, which is precisely the
~0.4/~0.8-bit terms from Section 2.

### Initialization: $B = 0$, $A$ random

LoRA initializes $A$ from a small random Gaussian and sets $B = 0$. The
immediate consequence is that $\Delta W = BA = 0$ at the start, so training
begins *exactly* at the pretrained function — no initial perturbation, just like
the adapter's near-identity start.

But why this *asymmetric* choice, rather than zeroing both or zeroing $A$? It is
a symmetry-breaking argument. Consider the first gradient step. The gradient
with respect to $B$ is proportional to $(\text{upstream gradient}) \cdot
(Ax)^\top$; since $A$ is random and nonzero, $Ax \ne 0$, so $B$ receives a
nonzero gradient and starts to move immediately — even though $B$ was zero, its
*gradient* is not. The gradient with respect to $A$ is proportional to $B^\top
\cdot (\text{upstream gradient})$; with $B = 0$ this is zero on the very first
step, so $A$ holds still initially, then begins updating once $B$ has moved off
zero. The net effect: the network starts undisturbed (good for stability), yet
both factors are trainable (gradients flow). Zeroing *both* would leave the
update permanently stuck at zero with no gradient to break the symmetry; the
$B=0$, $A$-random choice gives you the clean start without the dead gradient.

### The scaling factor $\alpha / r$

LoRA does not add $BA$ raw; it adds a scaled version:

$$
h = W_0 x + \frac{\alpha}{r} \, BA \, x.
$$

The reason is to **decouple the effective magnitude of the update from the
choice of rank**. As you increase $r$, you sum more rank-1 outer products into
$BA$, which tends to change the magnitude of the update — so a learning rate
tuned at one rank would be wrong at another. Dividing by $r$ counteracts that
scaling, and multiplying by a constant $\alpha$ gives a single gain knob.
*Precisely:* the $\alpha/r$ factor makes the optimization roughly invariant to
$r$, so you can sweep the rank without re-tuning the learning rate; $\alpha$ is
then a separate, stable hyperparameter (commonly fixed at $\alpha = r$ or
$\alpha = 2r$) that controls how strongly the adaptation is applied.

### Where LoRA is applied

LoRA is applied to selected weight matrices, most commonly the **attention
projection matrices** — the query and value projections $W_q$ and $W_v$, and
sometimes $W_k$, the output projection $W_o$, and the MLP weights. The original
work found that adapting $W_q$ and $W_v$ alone already captures most of the
benefit. The intuition is that attention projections are where the model decides
*what to attend to and what to read out* — the routing of information — and that
task specialization concentrates there, giving high quality per trainable
parameter. You do not need to LoRA-fy every matrix; you place adapters where the
adaptation signal is richest.

### Benefits and limitations

The deck's benefits map onto everything above. **Parameter efficiency**: train a
fraction of a percent of the weights. **Efficient storage**: a saved task is
just its $B$ and $A$, a few megabytes, against the gigabytes of a full
fine-tuned copy. **Lower memory footprint**: no gradients or optimizer state for
the frozen base (the Section 2 result). **Task-specific adaptation**: keep one
base and many small adapters. **Comparable results**: rank-$r$ updates match
full fine-tuning on a wide range of tasks. And the property that distinguishes
LoRA from adapters specifically: at inference you can **merge** the update,
computing $W = W_0 + \frac{\alpha}{r}BA$ once and folding it into the original
matrix. The served model has the *identical architecture and identical latency*
as the base — LoRA pays nothing at inference, fixing adapters' core weakness.

The limitations are the price of the low-rank constraint. **Fine-tuning scope**:
a rank-$r$ update cannot express a change that genuinely needs many independent
directions; tasks demanding large or high-rank shifts in behavior can exceed what
a small $r$ can represent, so LoRA can underperform full fine-tuning in those
regimes. **Hyperparameter optimization**: you now have to choose $r$, $\alpha$,
and which matrices to adapt — real tuning burden that full fine-tuning, for all
its cost, does not impose.

---

## 6. Q-LoRA — Quantized LoRA

### Why it exists

Section 2 showed the residual problem LoRA leaves: after freezing kills the
optimizer overhead, the **16-bit frozen base weights** become the dominant
memory cost (16 of LoRA's 17.6 bits). Q-LoRA (Dettmers et al., 2023) attacks
exactly that term. The base is frozen anyway — you never update it — so why
store it at full precision? Q-LoRA quantizes the frozen base to **4 bits** and
keeps the trainable LoRA adapters at higher precision. It is LoRA plus
quantization of the part LoRA already decided not to train.

### Quantization, and the absmax method

Quantization maps high-precision floating-point values onto a small grid of
integers, storing the integers plus a scale factor and dequantizing on demand
for computation. The deck's pedagogical entry point is **absolute-maximum
(absmax)** quantization, which maps an FP32 tensor into the signed int8 range
$[-127, 127]$.

The recipe: find the largest absolute value in the tensor, and set the scale so
that value maps to 127.

$$
\text{scale} = \frac{127}{\max_i |x_i|}, \qquad x^{\text{int}}_i =
\operatorname{round}(\text{scale} \cdot x_i).
$$

Work the deck's example. Given the FP32 tensor $[1.2,\ -3.1,\ 0.8,\ 2.4,\ 5.4]$,
the absolute maximum is $5.4$, so the **quantization constant** is

$$
\text{scale} = \frac{127}{5.4} \approx 23.5.
$$

Multiply and round each entry:

- $1.2 \times 23.5 = 28.2 \rightarrow 28$
- $-3.1 \times 23.5 = -72.85 \rightarrow -73$
- $0.8 \times 23.5 = 18.8 \rightarrow 19$
- $2.4 \times 23.5 = 56.4 \rightarrow 56$
- $5.4 \times 23.5 = 126.9 \rightarrow 127$

giving the int8 tensor $[28,\ -73,\ 19,\ 56,\ 127]$, exactly as on the slide.
The largest-magnitude entry lands on 127 by construction, which is the point of
absmax: it uses the full integer range without clipping. To **dequantize**, you
divide by the scale: $28 / 23.5 \approx 1.191$, recovering $1.2$ to within
rounding error. That small gap is the quantization error, and it is the price of
the compression.

### Why this saves memory — and why quality survives

Storage drops mechanically: int8 is 1 byte versus FP32's 4 (a 4× saving), and
Q-LoRA's actual scheme uses 4-bit weights, an 8× saving over FP32 and 4× over
the 16-bit base. That is precisely the "weight: 4 bits" line that took the 65B
footprint from 143 GB to 42 GB and from 4 GPUs to 1.

The deeper question is why such aggressive compression of the base does not wreck
the model. Two reasons. First, the quantization is applied only to the **frozen**
base; the **trainable** LoRA adapters stay at full precision, so the part of the
model that is actually being optimized is never degraded. During the backward
pass, the 4-bit base weights are dequantized so that gradients can flow through
them *into* the adapters — the base provides an accurate-enough reference for the
adapters to learn against, while only the adapters absorb the precision they
need. Second, the deck's MMLU-on-FLAN-v2 results make the empirical case: across
Llama-7B, 13B, 33B, and 65B, 4-bit schemes (Float4 and especially **NF4**) track
the 16-bit **BFloat16** baseline closely, with essentially no accuracy loss on
average. Four-bit storage of the base is, in practice, nearly free in quality.

### Beyond absmax: the real Q-LoRA recipe

The slides teach absmax as the intuition; the production method refines it in
three ways worth knowing:

- **NF4 (4-bit NormalFloat).** Pretrained weights are approximately
  zero-mean Gaussian, so a *uniform* grid (which absmax produces) wastes
  resolution. NF4 instead places its quantization levels at the quantiles of a
  normal distribution, so the bins are dense where weights are common and sparse
  where they are rare — an information-theoretically near-optimal grid for
  normally distributed data. This is why NF4 edges out plain Float4 in the
  deck's chart.
- **Double quantization.** The scale factors (quantization constants)
  themselves take space when computed blockwise; Q-LoRA quantizes *those*,
  shaving a little more memory.
- **Paged optimizers.** Optimizer-state memory spikes are offloaded to CPU
  memory to avoid out-of-memory crashes, smoothing the single-GPU budget.

### Connection

Q-LoRA is a strict composition: **LoRA's frozen, low-rank-update structure, with
the frozen part stored in 4 bits.** It inherits everything LoRA gives — mergeable
updates, small per-task storage, no forgetting — and adds the base-weight
compression that LoRA's own arithmetic showed was the next thing to fix. The
arc from full fine-tuning to LoRA to Q-LoRA is one continuous descent of the
dominant cost term: optimizer overhead, then base-weight precision.

---

## 7. DoRA — Weight-Decomposed Low-Rank Adaptation

### The specific weakness of LoRA it addresses

LoRA is cheap and effective, but it is not full fine-tuning, and the *way* it
falls short turns out to be specific and diagnosable. DoRA (Liu et al., 2024)
starts from an analysis: decompose any weight matrix into a **magnitude** (how
large each column is) and a **direction** (which way each column points, as a
unit vector), and then watch how these two components move during training. The
finding is that **full fine-tuning and LoRA produce characteristically different
magnitude–direction update patterns.** Full fine-tuning can adjust a column's
magnitude and its direction more or less independently. LoRA, applying a single
coupled low-rank update, tends to entangle the two — it cannot freely change
magnitude without dragging direction along, and vice versa. That coupling limits
how closely LoRA can imitate full fine-tuning's learning behavior, and it shows
up as a quality gap, especially at low rank. DoRA's fix is to **decouple
magnitude and direction and adapt them separately.**

### The decomposition

Following the deck's diagram, write a pretrained weight matrix $W_0 \in
\mathbb{R}^{d \times k}$ as a magnitude vector times a normalized direction
matrix, column by column:

$$
W_0 = m \cdot \frac{V}{\lVert V \rVert_c}, \qquad m = \lVert W_0 \rVert_c \in
\mathbb{R}^{1 \times k}, \quad V = W_0 \in \mathbb{R}^{d \times k},
$$

where $\lVert \cdot \rVert_c$ is the **column-wise** vector norm. Each column of
$V/\lVert V\rVert_c$ is a unit vector (the direction), and the corresponding
entry of $m$ scales it back to its original length (the magnitude). At
initialization this is an exact, lossless rewrite of $W_0$.

### How DoRA trains

DoRA now treats the two components differently, which is the entire idea:

- **Direction** gets the LoRA treatment. The directional component is
  $V + \Delta V$, where the base $V = W_0$ is frozen and the update $\Delta V =
  BA$ is the familiar low-rank, trainable LoRA pair. So the *direction* is
  adapted parameter-efficiently, exactly as in Section 5.
- **Magnitude** is trained directly and independently. The vector $m$ is a
  trainable parameter in its own right — just one scalar per column, $k$
  parameters total, which is negligible — and it is *not* forced through the
  low-rank bottleneck.

The merged, adapted weight is then renormalized so that $m$ continues to mean
"magnitude" after the directional update:

$$
W' = m \cdot \frac{V + \Delta V}{\lVert V + \Delta V \rVert_c} = m \cdot
\frac{W_0 + BA}{\lVert W_0 + BA \rVert_c}.
$$

The renormalization is what keeps the decomposition honest: $BA$ is allowed to
change the *direction* of each column, the division by $\lVert V + \Delta V
\rVert_c$ strips out whatever magnitude change that update incidentally
introduced, and then the separately learned $m$ sets the magnitude on purpose.
Magnitude and direction are now controlled by different, independent parameters —
recovering the flexibility LoRA's coupling denied, at the cost of only the tiny
$m$ vector.

### Connection

DoRA is **LoRA applied to the direction, plus a cheap independent magnitude.**
It is the most surgical method in the lecture: it does not change the cost
profile much (the extra $m$ is one scalar per column) and it does not abandon
LoRA — it wraps LoRA to repair one identified defect, the magnitude–direction
coupling, and as a result tracks full fine-tuning's learning dynamics more
closely and beats LoRA at equal rank. Like LoRA, the whole thing can be merged
into $W'$ at inference, so there is no latency penalty.

---

## 8. Fine-Tuning with Multiple Adapters

### The payoff of everything before it

The deck's diagram is a single frozen pretrained LLM with a fan of adapters
hanging off it — summarization, proofreading, mail replies, tone adjustment,
refining, query handling, friendliness, urgency detection, sentiment analysis,
translation, paraphrasing, fact-checking, code generation, text completion. This
slide is the cash-out of the entire quality-versus-cost story.

Here is why it works, and why it was impossible under full fine-tuning. With
full fine-tuning, each task produces a *complete, independent copy* of the
65-billion-parameter model — gigabytes per task, and a separate model to load
and serve for every behavior. That does not scale past a handful of tasks. With
PEFT, the base is **frozen and shared**, and each task's specialization is a tiny
adapter (a few megabytes of $B$ and $A$). So you load the base **once** and
**hot-swap** adapters per request: the same resident 65B model becomes a
summarizer, then a translator, then a code generator, by attaching a different
few-megabyte module. Dozens or hundreds of specializations cost essentially
nothing in marginal storage and require no extra base copies.

### How multiple adapters are combined

Because LoRA-style updates are additive and mergeable, several composition
patterns become available:

- **Swapping.** Keep the frozen base resident and attach the relevant adapter
  per request — the simplest and most common pattern, and the one the diagram
  depicts.
- **Merging.** Combine multiple adapters into one effective update by weighted
  sum, $W = W_0 + \sum_i w_i B_i A_i$, to get a model that blends several task
  behaviors, or to bake a chosen adapter permanently into the base.
- **Routing and composition.** Serve many adapters concurrently and route each
  input to the right one (or a learned mixture), as in systems built to host
  large numbers of adapters behind a single shared base. This makes a single
  deployed model act as a whole portfolio of specialists.

### Connection

This is the deployment-level dividend of the lecture's central tradeoff.
Adapters and LoRA made specialization *cheap to train and store*; Q-LoRA made
the shared base *cheap to hold in memory*; DoRA made the specialization
*higher-quality at the same cost*. Put together, they let one frozen (and
possibly 4-bit-quantized) base serve an arbitrary number of high-quality,
swappable task behaviors at near-zero marginal cost — the quality-versus-compute
tradeoff resolved so favorably that breadth of capability becomes nearly free.

---

## Coda — The Rest of the Deck

The lecture closes with three topics outside the PEFT arc but worth placing on
the same map of quality-versus-compute decisions.

**Fine-tuning vs. RAG.** Fine-tuning bakes behavior and knowledge *into the
weights*; retrieval-augmented generation (Gao et al., 2023) leaves the weights
alone and injects knowledge *at inference time* by retrieving relevant documents
into the context. The tradeoff is structural. Fine-tuning is the right tool when
the need is **behavioral** — a task skill, an output format, a house style, low
latency, or internalized domain fluency — and when the underlying knowledge is
stable. RAG is the right tool when the need is **factual and changing** — large,
frequently updated, or proprietary knowledge that you want to edit without
retraining, with the bonus that grounding generations in retrieved text reduces
hallucination on facts. They are complementary, not competing: a common pattern
fine-tunes for behavior and uses RAG for knowledge.

**Continual learning.** Real deployments face a *stream* of new data and new
classes over time (the deck shows a model growing from recognizing a few classes
to many across successive versions). The goal is to keep adapting without
retraining from scratch, and the central obstacle is the **catastrophic
forgetting** introduced back in Section 1 — new learning overwriting old
competence. Methods include rehearsal/replay of old data, regularization that
penalizes moving weights important to past tasks, and parameter-isolation
approaches that give each task its own parameters — which is, not coincidentally,
exactly what per-task adapters from Section 8 provide. PEFT's freeze-the-base
guarantee is itself a continual-learning tool.

**Knowledge distillation.** Distillation transfers the competence of a large
**teacher** model into a smaller **student** by training the student to match the
teacher's *soft* outputs — its full probability distribution over tokens, not
just the hard top-1 answer. Those soft targets carry "dark knowledge": the
relative probabilities the teacher assigns to alternatives encode similarity
structure that a one-hot label throws away. The payoff is a cheaper model to
serve at a modest quality cost — another point on the same compute frontier,
trading a one-time teacher-distillation expense for permanently cheaper
inference. The deck's inclusion of a 2026 OSTP memo on "adversarial distillation"
flags the flip side: because a model's outputs leak its knowledge, distillation
is also a vector for *extracting* a proprietary model through its API — a
security and intellectual-property concern, not just an efficiency technique.

---

*Throughline, in one sentence: full fine-tuning pays gradient and optimizer cost
on every parameter; LoRA freezes the base to erase that overhead; Q-LoRA
quantizes the now-dominant frozen base; DoRA recovers the quality LoRA's
low-rank coupling left on the table; and multiple adapters turn the whole frozen
base into a near-free platform for many specialists — each step targeting exactly
the cost or quality gap the previous step exposed.*
