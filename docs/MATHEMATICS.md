# Dense MLP mathematics and glossary

This derivation describes `DenseNetwork`. Architecture-specific equations are in [the individual guides](README.md); the other differentiable models share [the reverse-mode training engine](TRAINING.md).

A **layer** contains neurons. A **weight** connects one neuron to an input from the previous layer. A **bias** shifts a neuron's weighted sum. An **activation** is the neuron's output. A **parameter** is a weight or bias. A **sample** is an input and target pair. An **epoch** visits every training sample once.

## Forward pass

Let `a⁰ = x` be the input vector. For each layer `l = 1 … L`:

```text
zˡ_j = bˡ_j + Σ_i wˡ_ji aˡ⁻¹_i
aˡ_j = σ(zˡ_j)
σ(z) = 1 / (1 + exp(-z))
```

Weights are stored by destination neuron, then source input. Bias follows that neuron's weights. Hidden and output activations are sigmoid, independently for each output. Outputs do not sum to one and are not calibrated probabilities.

For numerical stability the code uses `e = exp(-abs(z))`, returning `1/(1+e)` for nonnegative `z` and `e/(1+e)` otherwise. This prevents exponential overflow, although extreme activations still round to zero or one.

## Initialization

Every layer with `fanIn` inputs and `fanOut` outputs draws independent weights uniformly from:

```text
[-sqrt(6 / (fanIn + fanOut)), +sqrt(6 / (fanIn + fanOut)))
```

Biases start at zero. This is Xavier/Glorot uniform initialization. A fresh `System.Random(seed)` belongs to each constructor; training does not consume random numbers. Identical seeds, topology and runtime reproduce initialization and training with identical sample order.

## Loss and backpropagation

For one sample, with target `t`:

```text
E = ½ Σ_j (aᴸ_j - t_j)²
σ′(z) = σ(z) (1 - σ(z))
δᴸ_j = (aᴸ_j - t_j) aᴸ_j (1 - aᴸ_j)
δˡ_i = (Σ_j wˡ⁺¹_ji δˡ⁺¹_j) aˡ_i (1 - aˡ_i)
∂E/∂wˡ_ji = δˡ_j aˡ⁻¹_i
∂E/∂bˡ_j = δˡ_j
```

The loss is a **sum across outputs**, not a mean. The XOR example reports the arithmetic mean of these losses across its four samples, recomputed using the current network. Every delta uses the old weights. Only after all gradients are computed does a training step replace parameters:

```text
θ_new = θ_old - learningRate × ∇E
```

A larger output count changes the scale of the summed objective. There is no regularization term, momentum or adaptive learning rate.

## Numerical verification

The tests perturb each parameter `θ_i` by `h = 1e-5` and compare backpropagation against the central difference:

```text
∂E/∂θ_i ≈ (E(θ + h e_i) - E(θ - h e_i)) / (2h)
```

The absolute tolerance is `1e-8` for the moderate, seeded inputs used by the tests. Finite differences are an independent check of the gradient, not a training algorithm. A separate known-value test checks the forward pass and loss; learning tests check descent and XOR output behavior.
