# Validation scope

The package-free executable suite returns a failing exit code on assertion failures. Run it with:

```sh
dotnet build --configuration Release
dotnet run --project tests/NeuralNetworks.Tests --configuration Release --no-build
```

`dotnet test` does not run this suite. CI also runs both original XOR modes, every architecture demo and the formatter on Linux, macOS and Windows.

## Numerical behavior

- **Dense MLP:** known forward/loss values, independent multilayer arithmetic, extreme sigmoid saturation, numerical gradients through multiple hidden layers and multiple outputs.
- **RBF and autoencoder:** known Gaussian/head and encoding/decoding values, gradients through every trainable head or encoder/decoder parameter.
- **CNN:** independent known patch/pooling values, positive/negative ReLU branches and numerical gradients for shared kernels/head across several patch positions.
- **RNN, LSTM, GRU:** independent hand-computed two-step state/gate examples; numerical gradients through shared parameters across the unrolled sequence.
- **Transformer:** independent scaled dot-product scores and normalized attention values, row sums, other-position influence, detached attention arrays; numerical gradients through every projection, head and normalization gamma/beta.

For the tape models, central finite differences perturb each trainable parameter by `h = 1e-5` and compare with an absolute tolerance of `2e-6`. The dense test uses `1e-8`. Inputs are deliberately moderate and avoid ReLU kinks. Gradients and finite differences can both be misleading around a nondifferentiable point; this suite does not claim all-input numerical equivalence.

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

Gradient demos require mean loss below one fifth of initial loss and every output within 0.2 of its target. The original XOR example has its stricter 0.08 output threshold. These thresholds concern training fixtures, not held-out generalization. A failing CNN fixture with uncentered inputs exposed inactive ReLU filters; the guide records this limitation, and the executable example uses centered inputs.

## Boundary and ownership behavior

Tests reject null/wrong-length/non-finite vectors, invalid topology/kernel/center/bottleneck settings, invalid labels/targets/rates, untrained Hopfield recall, and non-bipolar memories. Parameter/center/output snapshots are copied. Seeded models produce identical predictions and training updates on the same runtime. Failed fitting/updates preserve previous state. Extremely large finite parameters/inputs exercise arithmetic overflow rejection.

## Limits of this evidence

There is no claimed line/branch coverage percentage, CRAP score, duplication score or comprehensive mutation score. Installed `crapper`, `dryer` and `mutator` do not support C#. Targeted manual mutation experiments, when performed, are recorded separately in [WORK_LOG.md](WORK_LOG.md).

The suite does not establish real-world model quality, large-scale performance, robustness to arbitrary corruption, long-sequence learning, multicore safety, serialized-model compatibility or convergence for all seeds/hyperparameters. Production benchmarking, domain datasets and independent evaluation remain outside this educational repository.
