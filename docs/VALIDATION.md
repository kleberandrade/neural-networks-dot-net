# Validation scope

The package-free executable suite returns a failing exit code on assertion failures. Run it with:

```sh
dotnet build --configuration Release
dotnet run --project tests/NeuralNetworks.Tests --configuration Release --no-build
```

`dotnet test` does not run this suite. CI also runs both original XOR modes, every architecture demo, the multi-head/encoder-decoder example, the two-layer text-generation example, file-backed corpus/checkpoint demo, minibatch Adam/resume demo, clipping/schedule demo and the formatter on Linux, macOS and Windows.

## Numerical behavior

- **Dense MLP:** known forward/loss values, independent multilayer arithmetic, extreme sigmoid saturation, numerical gradients through multiple hidden layers and multiple outputs.
- **RBF and autoencoder:** known Gaussian/head and encoding/decoding values, gradients through every trainable head or encoder/decoder parameter.
- **CNN:** independent known patch/pooling values, positive/negative ReLU branches and numerical gradients for shared kernels/head across several patch positions.
- **RNN, LSTM, GRU:** independent hand-computed two-step state/gate examples; numerical gradients through shared parameters across the unrolled sequence.
- **Transformer encoder:** independent single/multi-head scores, head-width scaling, concatenated-context outputs, row sums, other-position influence and detached attention arrays; numerical gradients through every projection, head and normalization gamma/beta.

- **Encoder-decoder:** every trainable parameter against numerical gradients, including masked objectives; future decoder inputs cannot change earlier predictions; padded features/targets cannot alter valid predictions, loss or gradients; masked attention weights are zero; cross-attention depends on source memory. Checks also cover unequal source/target lengths, multiple features, mask validation, repeatability and atomic overflow rejection.

- **Stacked/text Transformer:** numerical gradients through two encoder and two decoder layers and learned embeddings; nonzero signal in every layer; deterministic Unicode vocabularies, emoji roundtrips and invalid UTF-16; known uniform categorical loss/derivatives and extreme-logit stability; causal attention at every decoder layer, source dependence, PAD invariance, detached snapshots and invalid token/target/depth rejection. Greedy generation tests EOS termination, length caps, exclusion of structural tokens and ranking underflow. The trained text test requires exact greedy replies with EOS for two synthetic prompts.

- **Corpus/checkpoints/sampling:** strict UTF-8/JSONL and Unicode windows, duplicate/unknown fields and limits; complete checkpoint roundtrips including exact double bits, parameters, tokenizer, loss, generation and further SGD; malformed/truncated/corrupt/unsupported/oversized checkpoints; resumed shuffled epochs versus uninterrupted training; rejection of invalid late epoch samples before mutation; seeded top-k/temperature selection, caps, ties and structural-token exclusion.

For the tape models, central finite differences perturb each trainable parameter by `h = 1e-5` and compare with an absolute tolerance of `2e-6`. The dense test uses `1e-8`; the extended multi-head/encoder-decoder tests use `3e-6`; stacked numerical/token-model checks use `5e-6`. Inputs are deliberately moderate and avoid ReLU kinks. Gradients and finite differences can both be misleading around a nondifferentiable point; this suite does not claim all-input numerical equivalence.

## Learning behavior

The fixtures use deterministic synthetic data and fixed seeds, with no external datasets or services:

- Perceptron learns every AND case exactly.
- Dense MLP and RBF learn XOR.
- Hopfield corrects one flipped bit in an eight-bit memory and decreases energy.
- SOM tests exact neighborhood updates; its runnable demo also lowers two-cluster quantization error.
- CNN distinguishes two centered constant-intensity images.
- RNN, LSTM and GRU remember a first-step bit after two zero-input steps.
- Autoencoder reconstructs two complementary patterns through one latent unit.
- Transformer encoder distinguishes two small numerical sequences.
- A character-token model with two encoder/two decoder layers memorizes `a → ola` and `b → oi`, including EOS. Its mean token cross-entropy must fall below one tenth of its initial value; outputs must match both training replies exactly.
- A two-head encoder-decoder learns two-bit reversal using teacher forcing, then generates all four reversed sequences autoregressively without target inputs. Its target-length-two loss must fall below one tenth of the initial value and generated outputs must be within 0.2 of the reverse targets.

Gradient demos require mean loss below one fifth of initial loss and every output within 0.2 of its target. The original XOR example has its stricter 0.08 output threshold. These thresholds concern training fixtures, not held-out generalization. A failing CNN fixture with uncentered inputs exposed inactive ReLU filters; the guide records this limitation, and the executable example uses centered inputs.

The file-backed demo uses 20 original records with 17 distinct prompts and 16 replies, including a four-alternative prompt. Its gate requires mean sample loss below one fifth of initial loss, at least 80% of prompt groups producing a trained alternative with EOS, at least eight distinct matching replies, and exact pre/post-checkpoint generation equality. It prints actual counts rather than assuming all samples learned. These are memorization checks, not held-out accuracy.

## Boundary and ownership behavior

Tests reject null/wrong-length/non-finite vectors, invalid topology/kernel/center/bottleneck settings, invalid labels/targets/rates, untrained Hopfield recall, and non-bipolar memories. Parameter/center/output snapshots are copied. Seeded models produce identical predictions and training updates on the same runtime. Failed fitting/updates preserve previous state. Extremely large finite parameters/inputs exercise arithmetic overflow rejection.

## Limits of this evidence

There is no claimed line/branch coverage percentage, CRAP score, duplication score or comprehensive mutation score. Installed `crapper`, `dryer` and `mutator` do not support C#. Targeted manual mutation experiments, when performed, are recorded separately in [WORK_LOG.md](WORK_LOG.md).

The suite does not establish real-world model quality, large-scale performance, robustness to arbitrary corruption, long-sequence learning, multicore safety, serialized-model compatibility or convergence for all seeds/hyperparameters. Production benchmarking, domain datasets and independent evaluation remain outside this educational repository.


## Minibatch Adam

Four additional behavioral groups independently check two-step Adam arithmetic/moment correction, equal sample-gradient averaging at unchanged weights (including unequal target lengths), partial batches, transactional invalid/overflow rejection, model binding and detached optimizer snapshots. Saved/reloaded Adam epochs match uninterrupted weights and moment arrays exactly on the same runtime. Schema tests reject malformed dimensions, options, counters, missing moments and optimizer/progress mismatches even with valid recomputed checksums; a legacy v1 SGD envelope remains readable.

The Adam demo checkpoints/reloads at epoch 150 (step 750), resumes to epoch 300 (step 1,500), and checks the same loss/reply-diversity gates as the SGD corpus demo. All reported output measurements describe training-corpus memorization. Documentation diagrams are rendered and inspected separately; their arrows are explanatory and are not a performance benchmark.


## Clipping and learning-rate policy

Four additional groups check known global-norm clipping, finite extreme coordinates and small thresholds, detached arrays, warmup/decay/floor boundary values and invalid settings. Independent sample gradients establish that clipping applies after batch averaging and before moments; invalid/non-finite inputs and scheduled-rate underflow preserve parameters/state and schedule position. Interrupted/reloaded training crosses warmup/decay and matches uninterrupted weights, moments and counters exactly. Tests reject invalid/missing v3 policy fields with valid recomputed checksums and resume nonzero legacy v2 Adam state with clipping/scheduling disabled.

The scheduled demo uses the same learning/diversity/EOS gates as the corpus demo and an actual midpoint save/reload. Its training-corpus measurements do not establish superior generalization or speed. All examples run in the three-OS CI matrix.
