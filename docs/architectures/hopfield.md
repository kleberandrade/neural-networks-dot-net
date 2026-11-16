# Hopfield network

## How it works

A Hopfield network is an associative memory. Rather than mapping inputs to labels, it stores bipolar patterns and updates a noisy pattern toward a stable state. Every component must be `-1` or `+1`.

For `N` neurons and stored patterns `p`, Hebbian weights are:

```text
w_ij = (1/N) Σ_p p_i p_j, for i ≠ j
w_ii = 0
```

Weights are symmetric. Recall visits neurons one at a time, in index order, setting each to the sign of its current weighted field. An exactly zero field preserves its old value. This asynchronous update rule cannot increase the energy `E = -½ Σ_ij w_ij s_i s_j`; energy descent does not guarantee recovery of the desired memory.

## How it was programmed

[HopfieldNetwork.cs](../../src/NeuralNetworks/HopfieldNetwork.cs) builds a new symmetric weight array in `Fit` after validating the entire memory bank. Fitting replaces previous memories. Recall copies the query and updates the copy in place; later neurons see earlier updates in the same sweep. A sweep with no changes reports `Converged = true` in `HopfieldRecall`. Hitting `maxSweeps` first reports false; the returned pattern can be examined or recalled again.

## How to use it

```csharp
using NeuralNetworks;

var model = new HopfieldNetwork(size: 8);
model.Fit([[-1, -1, -1, -1, 1, 1, 1, 1]]);
var result = model.Recall([-1, 1, -1, -1, 1, 1, 1, 1], maxSweeps: 100);
var energy = model.Energy(result.Pattern);
```

Run `dotnet run --project examples/Architectures -- hopfield`. It flips one bit, recovers the stored eight-bit pattern, checks convergence and prints energy changing from `-1.75` to `-3.5`.

## Validation and limitations

Tests check noisy recall, fixed-point recall, known energy, fit-before-use, bipolar validation and preservation of memories after rejected fitting. Arrays passed to fitting are not retained.

This is the classical discrete network, not a modern continuous Hopfield layer. Capacity is limited; correlated patterns and spurious attractors can prevent correct recall. No arbitrary noise-correction guarantee, learning rate, gradient API, probability output or trainable biases are provided. A final state reached on the last permitted sweep may require one extra unchanged sweep to report convergence.

Further reading: [Hopfield's original paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC346238/).

Next: [SOM](som.md) · [all architecture guides](../README.md).
