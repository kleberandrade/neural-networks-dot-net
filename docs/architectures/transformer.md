# Transformer encoder

## How it works

Self-attention lets each position combine information from every position in the same input sequence. This model implements one bidirectional encoder block with one attention head:

```text
X = input projection + sinusoidal position encoding
Q = X W_Q + b_Q; K = X W_K + b_K; V = X W_V + b_V
A = softmax(Q Kᵀ / sqrt(modelSize)), across key positions per query
H = LayerNorm(X + AttentionOutput(A V))
Z = LayerNorm(H + FFN(H))
prediction = sigmoid(OutputHead(mean_over_positions(Z)))
```

The FFN is a linear block, ReLU, then another linear block, applied independently to each position. Normalization is over features at each position, with learned gamma/beta and epsilon `1e-5`. The two normalizations follow their residual additions (post-norm).

## How it was programmed

[TransformerEncoder.cs](../../src/NeuralNetworks/TransformerEncoder.cs) projects numerical features, adds sine/cosine positions, computes Q/K/V arrays, attention scores and normalized exponentials, then builds residual/FFN outputs. Softmax subtracts the maximum score before exponentiation for stability; every attention row sums to one. Mean pooling feeds a sigmoid prediction head. [LayerNormalization.cs](../../src/NeuralNetworks/Internal/LayerNormalization.cs) computes feature mean/variance and learned affine scaling. The [tape](../TRAINING.md) propagates gradients through attention, normalization and all projections.

Position `t`, feature pair `i` uses angle `t / 10000^(2i/modelSize)`, with sine for even feature indices and cosine for odd ones. Inputs are finite numerical features in time-major order. The model does not interpret strings or token IDs. `GetAttentionWeights(input)` returns a detached square array indexed by query position, then key position, so you can inspect the learned attention rows.

## How to use it

```csharp
using NeuralNetworks;

var model = new TransformerEncoder(sequenceLength: 2, inputSize: 1,
    modelSize: 4, outputSize: 1, feedForwardSize: 4, seed: 42);
double[][] sequences = [[0, 0], [1, 1]];
double[][] targets = [[0], [1]];
for (var epoch = 0; epoch < 1200; epoch++)
    for (var i = 0; i < sequences.Length; i++)
        model.TrainSample(sequences[i], targets[i], 0.5);
var output = model.Predict([1, 1]);
```

Run `dotnet run --project examples/Architectures -- transformer`. It distinguishes two tiny numerical sequences and checks a loss decrease from about `0.133329` to `0.000021`. The objective is supervised sequence classification, not next-token prediction.

## Validation and limitations

Tests compare every trainable projection, head and normalization parameter against central differences; check small-task learning, determinism, state isolation and invalid/overflowing calls. `modelSize` must be at least two, and all other dimensions positive.

This is **one single-head encoder block**, not the full encoder-decoder architecture or a language model. It has no multi-head attention, causal/padding masks, decoder, tokenizer, learned vocabulary embeddings, dropout, pretrained weights, generation or variable sequence lengths. Attention compares every pair of positions, so its score work grows quadratically with sequence length; a scalar tape adds substantial memory/allocation overhead. Only tiny examples are appropriate.

Further reading: [Attention Is All You Need](https://arxiv.org/abs/1706.03762). The original paper describes a larger multi-head encoder-decoder; this implementation deliberately exposes a compact encoder subset.

[All architecture guides](../README.md) · [shared training engine](../TRAINING.md).
