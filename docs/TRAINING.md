# Training engine and common contracts

## Two ways to see backpropagation

`DenseNetwork` keeps the explicit layer-by-layer derivative calculation described in [MATHEMATICS.md](MATHEMATICS.md). The newer differentiable models inherit `GradientNetwork`, which evaluates their scalar operations on a small reverse-mode differentiation tape. Both numerical model paths implement `IGradientNetwork` and use the same half sum of squared errors:

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
- All models implementing `IGradientNetwork` use sigmoid heads and targets in `[0, 1]`. Outputs are independent, not a softmax class distribution. RBF centers/width are fixed hyperparameters.
- A learning rate must be finite and positive. It has no convergence guarantee.
- Non-finite sums, products, tape values, derivatives or candidate updates throw `ArithmeticException`. Invalid arguments throw `ArgumentException` or a more specific subtype. Failed gradient/perceptron/SOM updates leave parameters unchanged; failed Hopfield fitting preserves the previous memory bank.
- Evaluation is stateless across calls. Recurrent state starts at zero for every sample; RNN/LSTM/GRU have no streaming or masking API. TransformerSequenceToSequence supports explicit source/target padding masks and causal decoder attention, with a fresh graph and copied masks per call.

## Flattened parameter ordering

A linear block stores destination neurons in order; each neuron has its input weights followed by its bias. Blocks are concatenated in this order:

- **MLP:** input-to-output layer order.
- **RBF:** output head only. Centers and width do not appear in `GetParameters`.
- **Autoencoder:** encoder, then decoder.
- **CNN:** kernels, then output head. Each filter's kernel weights are row-major, followed by its bias.
- **RNN:** output head, then recurrent cell. The cell's input is `[currentFeatures, previousHidden]`.
- **LSTM:** output head, forget gate, input gate, candidate, output gate. Gate inputs are `[currentFeatures, previousHidden]`.
- **GRU:** output head, update gate, reset gate, candidate. Candidate input is `[currentFeatures, resetGate × previousHidden]`.
- **TransformerEncoder:** input projection, Q, K, V, attention output projection, feedforward input, feedforward output, final head, normalization 1, normalization 2. Each normalization stores the gamma vector then beta vector. Head count splits Q/K/V coordinates without changing parameter count.
- **TransformerSequenceToSequence:** see the [encoder-decoder guide](architectures/transformer-seq2seq.md#how-it-was-programmed) for its source/target, encoder, decoder and normalization block order.

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

## Masked sequence objectives

The encoder-decoder uses the same SGD engine with a per-call forward function and a Boolean output-loss mask. Target-padding positions are omitted from the half summed squared-error objective; the loss is not normalized by valid token count. The forward closure holds only validated, copied masks for that call, so there is no mutable model-wide mask context. The typed source/decoder overloads validate both vectors, masks and targets before committing updates. The inherited flat interface remains unmasked; see the [shape contract](architectures/transformer-seq2seq.md#api-and-shapes).

## Stacks and categorical token training

`ParameterizedModel` now owns seeded parameter allocation, copied snapshots, derivative extraction and atomic SGD updates. `GradientNetwork` retains its numerical sigmoid/MSE interface; `TextTransformer` inherits parameter ownership without implementing that different interface. The text model supplies integer IDs, learned source/target embeddings, per-position vocabulary logits and mean categorical cross-entropy, including EOS and excluding trailing PAD. Its target shifting is automatic. See [the complete guide](architectures/transformer-text.md).

`TransformerBuilder` allocates fresh parameters per layer; `TransformerStacks` sends source through every encoder and decoder state through every decoder. All decoder cross-attention uses the final encoder memory. Causal masks remain active at every decoder layer. The numerical classes default to one layer for source compatibility; the text model defaults to two encoder/two decoder layers. Depth, head count and the exact vocabulary mapping must accompany text parameter snapshots.

Text parameter order is source embedding matrix, target embedding matrix (both token-major), every encoder layer, every decoder layer, then vocabulary head. Encoder/decoder block order is documented in the architecture guides. The numerical pooled encoder preserves its original first-layer/head slots and appends additional layers afterward. The numerical sequence model allocates all encoders before all decoders and its final head. These models provide no checkpoint file format.
