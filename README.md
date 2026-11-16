# Neural Networks in C# / .NET

[![C# 14](https://img.shields.io/badge/C%23-14-239120)](https://learn.microsoft.com/en-us/dotnet/csharp/whats-new/csharp-14)
[![.NET 10 LTS](https://img.shields.io/badge/.NET-10%20LTS-512BD4)](https://dotnet.microsoft.com/en-us/download/dotnet/10.0)
[![MIT license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![CI](https://github.com/kleberandrade/neural-networks-dot-net/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/kleberandrade/neural-networks-dot-net/actions/workflows/ci.yml)

Eleven small neural-network architecture families implemented from scratch in C#/.NET: classical classifiers, associative memory, competitive learning, spatial models and sequence models. Each has working code, a runnable example and an educational guide. The default branch is `main`.

This is a learning project, with explicit loops and `double` arrays. It is not a production machine-learning framework.

## Supported networks and learning guides

Every guide explains **how the network works, how it was programmed, how to use it, and its tested limitations**:

- [Perceptron](docs/architectures/perceptron.md): binary linear classification with the mistake rule.
- [MLP / dense feedforward](docs/architectures/mlp.md): sigmoid layers with explicit backpropagation; learns XOR.
- [RBF](docs/architectures/rbf.md): fixed Gaussian centers and width with a trained sigmoid head.
- [Hopfield](docs/architectures/hopfield.md): bipolar associative memory with Hebbian fitting and asynchronous recall.
- [SOM](docs/architectures/som.md): unsupervised rectangular Kohonen map.
- [CNN](docs/architectures/cnn.md): single-channel valid 2D kernels, ReLU and global average pooling.
- [RNN](docs/architectures/rnn.md): tanh recurrence with full backpropagation through time.
- [LSTM](docs/architectures/lstm.md): forget/input/output gates and a separate cell state.
- [GRU](docs/architectures/gru.md): reset-before recurrent gating.
- [Autoencoder](docs/architectures/autoencoder.md): sigmoid bottleneck and reconstruction.
- [Transformer](docs/architectures/transformer.md): multi-head encoder plus an [encoder-decoder](docs/architectures/transformer-seq2seq.md) with causal attention, cross-attention, padding and autoregressive generation.

Start with the [learning path and documentation index](docs/README.md). [Common training contracts](docs/TRAINING.md) explain parameter ordering and the small reverse-mode tape used by the newer gradient models. The MLP keeps its separate, explicit backpropagation implementation. These are specific educational variants, not every architecture or feature of a production framework.

## Requirements

- A supported OS for the [.NET 10 SDK](https://dotnet.microsoft.com/en-us/download/dotnet/10.0).
- .NET **10 LTS** SDK, with the latest servicing update. `global.json` selects an installed stable 10.0 SDK using `latestFeature` roll-forward.
- Git to clone the repository. No GPU, Python, external datasets or NuGet packages are required.

.NET 10 LTS is supported through November 14, 2028 according to [Microsoft's support policy](https://dotnet.microsoft.com/en-us/platform/support/policy). Builds restore only SDK reference packs; `NuGet.Config` intentionally clears package sources. Install the SDK before building offline.

## Install and quick start

```sh
git clone https://github.com/kleberandrade/neural-networks-dot-net.git
cd neural-networks-dot-net
dotnet restore
dotnet build --configuration Release --no-restore
dotnet run --project examples/Architectures --configuration Release --no-build -- all
dotnet run --project examples/Transformer --configuration Release --no-build -- all
```

To run only the original XOR example, use `dotnet run --project examples/Xor -- train`. The seeded `2 → 4 → 1` network trains for 10,000 epochs, taking four SGD steps per epoch in a fixed order. A validated run with SDK 10.0.404 produced:

```text
Initial mean loss: 0.135352
Final mean loss:   0.000085
0 XOR 0 -> 0.009774 (expected 0)
0 XOR 1 -> 0.986943 (expected 1)
1 XOR 0 -> 0.986834 (expected 1)
1 XOR 1 -> 0.015576 (expected 0)
```

Small floating-point differences can occur across platforms or runtime versions. The example exits with code 1 if any output differs from its target by 0.08 or more. This check measures the four training cases, not generalization to an unseen dataset.

## Use the library

Reference the source project from your own .NET 10 application:

```sh
dotnet add path/to/YourApp.csproj reference src/NeuralNetworks/NeuralNetworks.csproj
```

```csharp
using NeuralNetworks;

var network = new DenseNetwork([2, 4, 1], seed: 42);
double[] input = [0, 1];
double[] target = [1];
var before = network.Loss(input, target);
var preUpdateLoss = network.TrainSample(input, target, learningRate: 0.5);
var prediction = network.Predict(input); // detached double[]
```

Call `TrainSample` repeatedly over your dataset. A single call performs one gradient step; it does not fit a dataset or guarantee convergence. This MLP uses sigmoid in every layer. The other gradient models have architecture-specific internal activations and a sigmoid head. They share `IGradientNetwork`; inputs must be finite and targets in `[0, 1]`. Perceptron, Hopfield and SOM have their own learning APIs, described in their guides.

## Examples

Run the whole tour with `dotnet run --project examples/Architectures -- all`, or choose `perceptron`, `mlp`, `rbf`, `hopfield`, `som`, `cnn`, `rnn`, `lstm`, `gru`, `autoencoder` or `transformer`. Each example checks an actual outcome and returns nonzero on failure. See [the example source](examples/Architectures/Program.cs).


- **Learning XOR:** `dotnet run --project examples/Xor -- train` trains from seeded random weights and prints mean loss before and after training.
- **Inference only:** `dotnet run --project examples/Xor -- infer` loads explicitly constructed illustrative XOR parameters and predicts without training. These parameters are not a saved training result.
- **Inspect gradients:** `ComputeGradient(input, target)` returns an independent gradient vector without updating the model. The tests compare it to central finite differences.
- **Parameter snapshots:** `GetParameters()` and `SetParameters(values)` copy arrays. See [the API guide](docs/API.md) for ordering and validation.

### Multi-head Transformer and encoder-decoder

```sh
dotnet run --project examples/Transformer -- all
# Select just encoder or seq2seq:
dotnet run --project examples/Transformer -- seq2seq
```

The new example trains a two-head encoder and an encoder-decoder that reverses two-bit numerical sequences. It verifies free-running generation using only source and BOS, separately from teacher-forced training. See the [multi-head guide](docs/architectures/transformer.md) and [encoder-decoder guide](docs/architectures/transformer-seq2seq.md) for equations, code structure, masks and usage.

## Tests and CI

```sh
dotnet build --configuration Release
dotnet run --project tests/NeuralNetworks.Tests --configuration Release --no-build
dotnet run --project examples/Xor --configuration Release --no-build -- train
dotnet run --project examples/Xor --configuration Release --no-build -- infer
dotnet run --project examples/Architectures --configuration Release --no-build -- all
dotnet run --project examples/Transformer --configuration Release --no-build -- all
dotnet format --verify-no-changes --no-restore
```

The test project is a package-free executable that prints each result and exits nonzero on failure. **`dotnet test` does not run this suite.** Tests cover known forward/convolution/gate/attention arithmetic; finite differences for every trainable parameter in every gradient architecture, including multiple outputs; loss reduction and small learning tasks; AND/XOR, Hopfield recall/energy and SOM neighborhoods; deterministic initialization, array ownership, invalid inputs and atomic overflow rejection. See [validation details](docs/VALIDATION.md).

[CI](.github/workflows/ci.yml) runs these commands on Linux, macOS and Windows. Local validation is recorded in [the implementation log](docs/WORK_LOG.md). The installed ub-mode quality tools do not support C#, so no automated CRAP, duplication, coverage or mutation score is reported.

## Design and limitations

See the [individual architecture guides](docs/README.md), [common contracts](docs/TRAINING.md), [dense mathematics](docs/MATHEMATICS.md) and [dense API](docs/API.md).

- The models implement the specific variants listed above. CNN has one input channel/stage. RNN/LSTM/GRU use fixed lengths and final outputs; the Transformer encoder-decoder produces a vector at every target position. Transformer supports one multi-head encoder block and one causal decoder block. Numerical sequence generation is supported; tokenization and vocabulary-based language modeling are outside this implementation.
- All gradient heads are sigmoid with half summed squared-error loss. RBF centers and width are fixed; Perceptron, Hopfield and SOM use their own learning rules.
- Sequential per-sample SGD only. Data preparation, normalization, shuffling, evaluation splits and stopping criteria belong to the caller. Seeds and learning rates affect convergence. Sigmoid can saturate, ReLU filters can become inactive, and sequence gradients can vanish or explode.
- Finite inputs can still overflow arithmetic. Rejected steps preserve model parameters. Instances require synchronization around concurrent mutation.
- Explicit scalar loops and temporary graphs favor readability over speed. There is no GPU execution, optimized tensor backend, minibatch optimizer, gradient clipping or model persistence format. Use tiny datasets and short sequences.
- Seeds reproduce results on the same runtime, not across all runtime versions. Large dimensions can exhaust memory; parameter-count overflow is checked.
- Example training errors are measurements on synthetic training cases, with no real-world accuracy, throughput or generalization guarantee.

## License

[MIT](LICENSE), copyright © 2018 Kléber de Oliveira Andrade. The original notice is preserved. The library and examples have no third-party runtime dependencies; .NET and GitHub Actions retain their own licenses.
