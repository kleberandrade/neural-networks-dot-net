# Training engine and common contracts

## Two ways to see backpropagation

`DenseNetwork` keeps the explicit layer-by-layer derivative calculation described in [MATHEMATICS.md](MATHEMATICS.md). The newer differentiable models inherit `GradientNetwork`, which evaluates their scalar operations on a small reverse-mode differentiation tape. Both implement `IGradientNetwork` and use the same half sum of squared errors:

```text
E = ½ Σ_j (prediction_j - target_j)²
parameter_new = parameter_old - learningRate × ∂E/∂parameter
```

This sum is over output features. Mean dataset loss is the arithmetic mean of individual `Loss` calls, recomputed after an epoch. `TrainSample` returns the loss before its own update.

The shared interface has `InputSize`, `OutputSize`, `ParameterCount`, `Predict`, `Loss`, `ComputeGradient`, `TrainSample`, `GetParameters` and `SetParameters`. Perceptron, SOM and Hopfield use different learning rules and deliberately do not implement this gradient interface.

## How the tape was programmed

Read [Scalar.cs](../src/NeuralNetworks/Internal/Scalar.cs), then [GradientNetwork.cs](../src/NeuralNetworks/GradientNetwork.cs).

1. Every input and parameter becomes a leaf node. Each arithmetic result appends a node with its numeric value, parent indices and local derivatives.
2. For `c = a × b`, store `∂c/∂a = b` and `∂c/∂b = a`. Sigmoid, tanh, exponential, square root and ReLU similarly store their local partial derivatives.
3. Seed the loss gradient with 1 and walk nodes in reverse creation order. Accumulate `parentGradient += childGradient × localDerivative`.
4. A shared weight can have many children: accumulation adds every spatial or temporal contribution. This is how convolution weight sharing and full backpropagation through time work.
5. Collect gradients at parameter leaves, compute every candidate SGD update, and replace the parameter array only if every calculation remains finite.

The tape is recreated for each call, with iterative reverse traversal. It is an internal scalar teaching tool, not a public tensor/autodiff framework. Prediction currently also creates a tape for simplicity. Runtime and allocation costs make it appropriate for tiny examples.

## Initialization and array ownership

Linear blocks use Xavier uniform weights with bounds `±sqrt(6/(fanIn+fanOut))` and zero biases. Layer normalization starts with gamma 1 and beta 0. Recurrent gate biases, including LSTM forget biases, start at zero. The same topology, seed, runtime, inputs and training order reproduce results. This is not a cross-runtime serialization guarantee.

Models copy topology, RBF centers and supplied parameter arrays. Predictions, gradients, encodings and snapshots are detached arrays. Caller-owned inputs must not change during a call. Model instances are mutable and require external synchronization around concurrent mutation; otherwise use separate instances. There is no persistence format, optimizer state, minibatching, clipping, scheduling, regularization or GPU execution.

## Public boundaries

- Dimensions must be positive. Products/counts use checked indexing arithmetic. Huge allocations can still exhaust memory.
- Inputs and parameters must have the exact required length and contain finite doubles.
- All gradient models use sigmoid heads and targets in `[0, 1]`. Outputs are independent, not a softmax class distribution. RBF centers/width are fixed hyperparameters.
- A learning rate must be finite and positive. It has no convergence guarantee.
- Non-finite sums, products, tape values, derivatives or candidate updates throw `ArithmeticException`. Invalid arguments throw `ArgumentException` or a more specific subtype. Failed gradient/perceptron/SOM updates leave parameters unchanged; failed Hopfield fitting preserves the previous memory bank.
- Evaluation is stateless across calls. Sequence state starts at zero for every sample; there is no streaming state, sequence masking or padding handling.

## Flattened parameter ordering

A linear block stores destination neurons in order; each neuron has its input weights followed by its bias. Blocks are concatenated in this order:

- **MLP:** input-to-output layer order.
- **RBF:** output head only. Centers and width do not appear in `GetParameters`.
- **Autoencoder:** encoder, then decoder.
- **CNN:** kernels, then output head. Each filter's kernel weights are row-major, followed by its bias.
- **RNN:** output head, then recurrent cell. The cell's input is `[currentFeatures, previousHidden]`.
- **LSTM:** output head, forget gate, input gate, candidate, output gate. Gate inputs are `[currentFeatures, previousHidden]`.
- **GRU:** output head, update gate, reset gate, candidate. Candidate input is `[currentFeatures, resetGate × previousHidden]`.
- **Transformer:** input projection, Q, K, V, attention output projection, feedforward input, feedforward output, final head, normalization 1, normalization 2. Each normalization stores the gamma vector then beta vector.

`SetParameters` validates vector length, not topology metadata. A snapshot belongs with its exact architecture, dimensions and hyperparameters; equal parameter counts do not establish compatibility.

## A minimal training loop

```csharp
using NeuralNetworks;

IGradientNetwork model = new RecurrentNetwork(3, 1, 3, 1, seed: 42);
double[][] inputs = [[0, 0, 0], [1, 0, 0]];
double[][] targets = [[0], [1]];
for (var epoch = 0; epoch < 1200; epoch++)
    for (var i = 0; i < inputs.Length; i++)
        model.TrainSample(inputs[i], targets[i], learningRate: 0.5);
var output = model.Predict([1, 0, 0]);
```

The executable tour uses this contract and checks actual loss reduction. Dataset preparation, normalization, shuffling, splits and stopping rules remain the caller's responsibility.
