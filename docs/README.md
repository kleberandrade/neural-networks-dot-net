# Learning guide

Start with **Perceptron → MLP → Autoencoder** to understand classification, backpropagation and representation learning. Then explore alternative learning rules with **RBF, SOM and Hopfield**. Finish with **CNN → RNN → LSTM → GRU → Transformer** to study spatial and sequential structure.

This repository implements eleven educational architecture families. Each guide explains the computation, maps it to the actual source, gives a runnable example, and states the implementation's boundaries. These are small working models, not every variant of each family.

## One guide per architecture

- [Perceptron](architectures/perceptron.md): a binary linear decision boundary and the mistake update.
- [Multilayer perceptron / dense feedforward network](architectures/mlp.md): nonlinear layers and explicit backpropagation.
- [Radial basis function network](architectures/rbf.md): Gaussian features with a learned output head.
- [Hopfield network](architectures/hopfield.md): bipolar associative memory and energy descent.
- [Self-organizing map](architectures/som.md): unsupervised competitive learning on a grid.
- [Convolutional neural network](architectures/cnn.md): shared spatial kernels and pooling.
- [Recurrent neural network](architectures/rnn.md): state shared through time.
- [Long short-term memory](architectures/lstm.md): recurrent gates and a separate cell state.
- [Gated recurrent unit](architectures/gru.md): update/reset gates and a single state.
- [Autoencoder](architectures/autoencoder.md): an encoder bottleneck and reconstruction.
- [Transformer encoder](architectures/transformer.md): attention, positions, residuals and normalization.

## Shared material

- [Installation and quick start](../README.md#install-and-quick-start).
- [Original dense API](API.md) and [dense mathematics](MATHEMATICS.md).
- [Training engine and common contracts](TRAINING.md): shared gradient API, reverse-mode differentiation, ownership, validation and parameter ordering.
- [Validation](VALIDATION.md): what the tests establish and what they do not establish.
- [Implementation log](WORK_LOG.md): actions, reasons and validation evidence.

## Run the complete tour

From the repository root with the .NET 10 SDK installed:

```sh
dotnet build --configuration Release
dotnet run --project examples/Architectures --configuration Release --no-build -- all
```

Use any individual selector: `perceptron`, `mlp`, `rbf`, `hopfield`, `som`, `cnn`, `rnn`, `lstm`, `gru`, `autoencoder`, or `transformer`. Invalid selectors return exit code 2; failed example acceptance checks return 1.

Every example uses synthetic data and runs offline. Reported training errors describe these examples only, without implying generalization or real-world performance.
