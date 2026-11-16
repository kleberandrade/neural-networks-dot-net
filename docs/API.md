# Dense MLP API guide

This page documents `DenseNetwork`. For the other architectures, see the [learning index](README.md) and [shared gradient contracts](TRAINING.md).

Namespace: `NeuralNetworks`. Type: `sealed class DenseNetwork`.

## Construction and dimensions

`new DenseNetwork(int[] layerSizes, int seed = 0)` copies the topology. At least two positive layer sizes are required. `[2, 4, 1]` means two inputs, four hidden neurons, one output; `[2, 1]` has no hidden layer. Arbitrarily many hidden layers and multiple outputs are supported, subject to memory and integer indexing limits.

`InputSize`, `OutputSize` and `ParameterCount` are read-only. Each connection layer has `fanOut * (fanIn + 1)` parameters.

## Evaluation

`double[] Predict(double[] input)` returns a new output array. The input length must equal `InputSize`. Inputs are unrestricted finite doubles; scaling is the caller's responsibility.

`double Loss(double[] input, double[] target)` returns half the sum of squared output errors. Target length must equal `OutputSize`, with each value in `[0, 1]`.

`double[] ComputeGradient(double[] input, double[] target)` computes the analytic loss gradient. The model does not change. The returned array is owned by the caller and uses the parameter order below.

## Training

`double TrainSample(double[] input, double[] target, double learningRate)` validates a finite, strictly positive learning rate, computes the forward pass and gradient, and applies one SGD update. It returns that sample's loss **before** updating the weights. To measure epoch loss, call `Loss` over the dataset after the epoch rather than averaging stale pre-update values.

Validation and all parameter calculations finish before replacement. Invalid arguments or detected non-finite arithmetic do not partially update a model. There is no dataset-level transaction: earlier successful calls remain applied if a later sample fails.

## Parameter ownership and ordering

`double[] GetParameters()` returns a detached snapshot. `void SetParameters(double[] values)` requires exactly `ParameterCount` finite values and copies them after validation. Predictions and gradients also return detached arrays. Caller-owned arguments must not be concurrently mutated during a call.

Flatten parameters by connection layer from input to output, then destination neuron, then its input weights, followed by its bias. For `[2, 2, 1]`:

```text
hidden neuron 0: w00, w01, b0
hidden neuron 1: w10, w11, b1
output neuron 0: v00, v01, c0
```

There are nine parameters total. Gradient entries match this exact sequence. Loading a snapshot requires a network with the **same topology**, not just the same parameter count; the vector contains no topology metadata. The inference example illustrates this ordering with fixed parameters.

## Exceptions

- `ArgumentNullException`: a required array is null.
- `ArgumentException`: invalid layer dimensions, wrong vector length, non-finite vector entries, or out-of-range target values.
- `ArgumentOutOfRangeException`: nonpositive or non-finite learning rate.
- `ArithmeticException`: non-finite intermediate sums, products, gradients or candidate updates.
- `OverflowException`: topology parameter counts exceed integer indexing limits.
- `OutOfMemoryException`: a requested topology exceeds available memory.

Invalid inputs are rejected at public boundaries. Finite input values do not guarantee finite intermediate arithmetic; scale values appropriately. The library deliberately rejects overflow even if a later sigmoid could hide it through saturation.

## Concurrency and persistence

Use one writer with external synchronization, or separate networks. This API offers no concurrent mutation guarantees and no model serialization format. For experiments, retain a topology and parameter snapshot together; do not treat `System.Random` seed behavior as a permanent model storage format.
