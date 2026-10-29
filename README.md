# RidgeFill: a training-free late-layer cache fill for faster long-prompt prefill

**Technical note, 2026-10-05.** Dylan Johnston, with Robert P Smith. A full preprint with locked-benchmark results follows;
this note records the method and the evidence that exists today, each result labelled with how strong it is.

## Summary

- **The idea.** For prompt tokens, run only the first half of the model. A closed-form map, fitted once from one
  pass over calibration text, predicts what the later layers would have received, and those layers write their
  caches from the prediction. The last few thousand prompt tokens and every decoding step still run the full
  model.
- **Engine result.** On Qwen3.8-Flash-Next in the radiance engine, the quality mode makes time to first token
  **2.10x faster at 32K tokens** with **no measurable perplexity change** (+0.1%, 95% CI −1.3% to +1.5%) and
  **72 of 72 needles retrieved**. The speed mode is 2.55x faster at +2.4% perplexity. Decode speed is unchanged.
- **No training.** The map is a ridge regression solved in closed form. The same projector file, fitted on one
  engine and quantisation, works unchanged on a second.
- **Why it works differently across architectures.** On a dense model without QK-norm (Llama-3.1-8B) the fitted map
  is what makes the fill usable, and the error that matters is in the keys. On a QK-normed hybrid (Qwen3.5-9B) the
  fitted map adds little, and the error that matters is in the values.

## Origin

The idea of serving late-layer caches from an earlier layer's state comes from YOCO (Sun et al., 2024,
arXiv:2405.05254) and its trained descendant, the Causal Encoder-Decoder in DeepSeek-V4.1-Flash (DeepSeek-AI,
2026, arXiv:2609.19969, section 2.2). There, the upper half of the layers take their K/V from a learned projection
of the middle layer's hidden state, and only the last window of tokens is replayed through the full stack. That
design is pretrained into the model.

RidgeFill retrofits the idea onto models that were not trained for it. The first public retrofit we know of is
kishida's [Q3-8B-KVA-Projector](https://huggingface.co/kishida/Q3-8B-KVA-Projector) (2026-09-10), which trains
per-layer K/V heads for Qwen3-8B. The name KVA is theirs; our engine code keeps it.

RidgeFill differs in four ways:
- it is fitted in closed form, not trained;
- it predicts each late layer's input, so the layer's own K/V projections (and, on hybrid models, its recurrent
  state update) run unchanged;
- it keeps an exact tail and can keep chosen prompt rows exact;
- it is measured on dense, hybrid and MoE models, including in serving engines.

## Method

- **Split.** Pick a split layer S. Prompt tokens run layers 0..S−1 exactly.
- **Fit.** One capture pass over calibration documents records, for every later layer, the hidden state at S and
  that layer's true input. A centred ridge regression per layer, with lambda chosen on held-out documents, gives
  the map. One capture serves every split.
- **Fill.** For the prompt prefix, the later layers skip their attention and MLP work. Each one writes its cache
  from the predicted input through its own projections: K/V on attention layers, the recurrent state on Gated
  DeltaNet layers.
- **Exact tail.** The last T prompt tokens (2,048 to 4,096) and all decode steps run every layer.
- **Corrections and exact rows** (used on Flash-Next):
  - a fitted per-layer constant added to each late GDN layer's recurrent state where the approximated stretch ends;
  - in quality mode, a share of rare tokens from three error-prone token classes (capitalised word starts,
    capitalised word pieces, mixed letter/digit/symbol tokens) runs exactly through the late layers, and the
    correction is scaled by how much of the recurrence was actually approximated.

## Results

Evidence tiers used below:
- **engine, measured**: run in a serving engine, timings measured;
- **dev, registered**: a prediction recorded in a hash-chained ledger before the run, on development prompts;
- **dev, exploratory**: development prompts, no prior registration.

### 1. Qwen3.8-Flash-Next in radiance (engine, measured; deterministic engine)

| | stock | quality mode | speed mode |
|---|---|---|---|
| time to first token, 16K prompt | 12,264 ms | 8,221 ms (1.49x) | 6,242 ms (1.97x) |
| time to first token, 32K prompt | 24,063 ms | 11,463 ms (2.10x) | 9,452 ms (2.55x) |
| perplexity vs exact | — | +0.1% [−1.3%, +1.5%] | +2.4% [+0.4%, +4.6%] |
| KL to exact / top-1 agreement | 0 / 100% | 0.037 / 91.3% | 0.067 / 89.4% |
| needles found (72 prompts) | 72 | 72 | 72 |
| decode, one user, settled | 8.18 ms/token | 7.99 ms/token | — |
| prefill cost on short prompts that are not approximated | — | +0.9% (2K) / +1.2% (8K) | same |

Setup:
- **Engine:** radiance 1.0.13 ([codeberg.org/StillDeadcode/radiance](https://codeberg.org/StillDeadcode/radiance))
  with its own Flash-Next deployment profile: two AMD R9700 cards, tensor parallel 2, FP8 KV cache, MTP depth 3,
  prefix cache on. Stock radiance is byte-identical run to run, so differences are not engine noise.
- **Model:** radiance's FP8/IQ4 build of Qwen3.8-Flash-Next
  (`StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe`).
- **Projector:** split 24 of 48, int8. It was fitted on Robert P Smith's MXFP4/FP8 checkpoint and ported unchanged; a
  refit on radiance's quantisation matched it within noise. Quality mode keeps 25% of class matches exact.
- **Tails:** quality T = 2,560, speed T = 2,048.
- **Timing:** median of 7 warmed requests, two rounds agreeing within 0.3%.
- **Perplexity:** last 512 tokens of 9 long documents (PG-19 windows of 8K, 16K and 32K tokens), paired against
  exact, bootstrap 95% interval.
- **Needles:** 72 chat prompts at 16K and 32K with one fact or the right one of four keys, hidden at 10–85% depth
  inside the approximated part, plus exact-tail controls.

### 2. Qwen3.8-Flash-Next in Robert P Smith's vLLM (engine, measured)

Speed-parity configuration (projector + GDN correction, split 24, T = 2,048): **+2.8% perplexity**
(+0.027 nats [+0.015, +0.039]) at **1.54x / 1.68x / 1.71x** faster time to first token at 9K / 16K / 32K.
Measured on a held-out tier of 19 PG-19 documents, with timings as the median of 5 runs. This engine is not
bit-deterministic before a planned top-k ordering fix, so differences under about 0.01 nats per document are
within its run noise.

### 3. Fitted map vs POP's identity fill (dev, registered)

POP (He et al., arXiv:2602.03295) fills skipped layers' caches from the split state unchanged: the identity map.
Our identity arm reproduces POP's fill exactly; it was checked against an implementation built from the paper,
with a largest difference of 3e-7. Here it runs at our split and tail, not POP's own configuration.

| model | perplexity-style loss, RidgeFill − identity | multi-key needle accuracy, RidgeFill − identity |
|---|---|---|
| Llama-3.1-8B (dense, no QK-norm) | −0.36 nats [−0.42, −0.30] | +0.28 [+0.21, +0.36] |
| Qwen3.5-9B (GDN hybrid, QK-norm) | −0.018 [−0.034, −0.004] | +0.11 [+0.03, +0.19] |

Pooled over three splits and two tails, on 23 graded development prompts and on the multi-key needle prompts whose
needle lies in the approximated rows. At split 8 on Llama, the identity fill retrieves 1 of 56 multi-key needles and RidgeFill 27.

### 4. Where the loss lives: keys vs values (dev, registered)

Oracle splices replace one half of the filled late-layer cache with the true values (Llama and Qwen3.5, split 12,
tail 1,024):

| | Llama-3.1-8B | Qwen3.5-9B |
|---|---|---|
| needles found, RidgeFill / exact keys only / exact values only / exact | 24 / 54 / 37 / 55 of 56 | 35 / 56 / 38 / 56 of 56 |
| which half costs more perplexity (exact keys − exact values) | keys: −0.065 [−0.093, −0.038] | values: +0.026 [+0.015, +0.038] |

- **Keys decide retrieval on both models.** Exact keys alone restore it.
- **The perplexity cost is in the keys on Llama and in the values on Qwen3.5.** Qwen3.5's QK-norm makes a key-size
  error invisible to attention, so only direction errors count there.
- **A deployable key-scale correction** measured on the exact tail removes 32% (split 12) and 54% (split 16) of
  RidgeFill's loss on Llama and does nothing on Qwen3.5.
- **POP's fill gets worse from split 12 to 16 on Llama** (+0.19 nats, 20 of 23 prompts) but not on Qwen3.5. In a
  registered probe on the two median prompts, exact keys remove 101–115% of that excess; exact values remove 1–5%,
  and the attention-sink row 0.4–2.5%.

## Limitations

- The reference-model results (sections 3–4) are on development prompts that also guided design choices. The
  frozen held-out sets (RULER, LongBench, a 126-document long-text set) have not been run yet.
- The radiance perplexity set is 9 documents and was also used during development. "No measurable change" is the
  right reading; equivalence to exact at a 1% margin is not shown.
- Every scored perplexity token sits after at least 2,048 exact tokens. The needle tests cover the approximated
  region directly.
- radiance timings were measured with display headroom reserved on one card. Headless speedups may be somewhat
  lower; not yet measured.
- Speed on the reference models is from a cost model, not measured. The measured speed results are both on
  Flash-Next.
- A trained map does better at the same split: SwiftKV's distilled maps beat the ridge fit on Llama when paired
  with our exact tail.

## What comes next

The preprint runs pre-registered final configurations once on the frozen held-out sets: RULER, LongBench and
long-document perplexity on Llama-3.1-8B, Qwen3.5-9B and Qwen3-8B. Qwen3-8B is dense with QK-norm, which separates
QK-norm from the hybrid design. It adds POP's own configuration as a baseline, measured dense-model prefill time,
and an expanded Flash-Next tier in Robert P Smith's engine with its deterministic fix. Code and projectors are released
with it.

## Credits

- **YOCO** and **DeepSeek-V4.1-Flash's Causal Encoder-Decoder** for the design; **kishida** for the first public
  retrofit.
- **Robert P Smith (tcclaviger):** the vLLM engine, its KVA integration, and co-authorship.
- **StillDeadcode:** radiance and the Flash-Next build it serves.

Cite as: Dylan Johnston and Robert P Smith, *RidgeFill: a training-free late-layer cache fill for faster long-prompt prefill*,
technical note, 2026-10-05.

---

Text and tables in this note: [CC BY 4.0](LICENSE).
