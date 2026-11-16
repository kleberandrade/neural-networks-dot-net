# Long short-term memory (LSTM)

## How it works

An LSTM maintains a cell state `c` as well as hidden state `h`. Gates control what to retain, add and expose. With `u_t = [x_t, h_(t-1)]`:

```text
f_t = sigmoid(W_f u_t + b_f)       forget gate
i_t = sigmoid(W_i u_t + b_i)       input gate
g_t = tanh(W_g u_t + b_g)          candidate
o_t = sigmoid(W_o u_t + b_o)       output gate
c_t = f_t × c_(t-1) + i_t × g_t
h_t = o_t × tanh(c_t)
y   = sigmoid(W_head h_T + b_head)
```

Multiplications between gate/state vectors are elementwise. The additive cell-state path offers a different route for gradient flow than a plain tanh recurrence. It does not guarantee learning arbitrarily long dependencies.

## How it was programmed

[LstmNetwork.cs](../../src/NeuralNetworks/LstmNetwork.cs) owns four linear gate blocks and returns new hidden and cell arrays at each step. [SequenceNetwork.cs](../../src/NeuralNetworks/SequenceNetwork.cs) supplies the loop and final head. All parameters are shared through time, and the [tape](../TRAINING.md) differentiates through both state paths. Gates use Xavier weights and zero biases, including the forget gate. This is a forget-gate LSTM without peepholes.

Both states reset to zero for every evaluation. Flat input order is time-major; for two features at each of three steps, supply six values. The base `InputSize` is the total flat length, while constructor `inputSize` is features per step.

## How to use it

```csharp
using NeuralNetworks;

var model = new LstmNetwork(sequenceLength: 3, inputSize: 1,
    hiddenSize: 3, outputSize: 1, seed: 42);
double[][] sequences = [[0, 0, 0], [1, 0, 0]];
double[][] targets = [[0], [1]];
for (var epoch = 0; epoch < 1200; epoch++)
    for (var i = 0; i < sequences.Length; i++)
        model.TrainSample(sequences[i], targets[i], 0.5);
var output = model.Predict([1, 0, 0]);
```

Run `dotnet run --project examples/Architectures -- lstm`. The tiny first-feature memory task lowers mean loss from about `0.125621` to `0.000356` with the supplied seed and settings.

## Validation and limitations

Tests verify a hand-computed two-step gate/cell result, every gate/head gradient against central differences, deterministic state reset, invalid dimensions/values, atomic overflow rejection and real learning.

Only one unidirectional recurrent layer and a final sigmoid head are implemented. There are no peepholes, projection layers, streaming API, masking, variable sequence lengths, sequence-to-sequence loss, clipping or optimized tensor kernels. Full BPTT keeps a graph for the whole sequence, so use small examples.

Further reading: [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735). The implementation uses the common explicit forget-gate equations above rather than attempting to reproduce every historical variant.

Next: [GRU](gru.md) · [all architecture guides](../README.md).
