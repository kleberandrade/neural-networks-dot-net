# Stacked Transformer and simple text generation

## How it works

This variant turns text into integer token IDs, looks up learned embeddings, runs configurable encoder/decoder stacks and predicts a vocabulary distribution at every decoder position. It is separate from the numerical sigmoid/MSE API: `TextTransformer` uses **mean categorical cross-entropy**.

```text
source characters -> IDs -> source embedding + positions
    -> encoder layer 0 -> … -> encoder layer (N-1) -> memory

[BOS, previous target IDs] -> target embedding + positions
    -> decoder layer 0 -> … -> decoder layer (M-1)
    -> vocabulary logits -> softmax probabilities
```

Each encoder layer has multi-head self-attention and an FFN. Each decoder layer has causal self-attention, cross-attention over the **final encoder memory**, and an FFN. Every sublayer has a residual connection and learned post-layer normalization. Different layers have independent weights; the same layer reuses its weights across positions. `modelSize` must divide evenly by `headCount`.

### Tokenization

`CharacterTokenizer.FromCorpus` fits an immutable vocabulary to supplied training text. The first four IDs are fixed:

- `0 = PAD`: trailing padding, ignored by the model.
- `1 = BOS`: beginning of decoder input.
- `2 = EOS`: end of the generated target.
- `3 = UNK`: a character absent from the fitted vocabulary.

Ordinary characters follow in ascending Unicode scalar-value order, independent of corpus order and culture. A token is a Unicode scalar (`Rune`), not a UTF-16 code unit or grapheme cluster. An emoji represented by a surrogate pair is one token; combining marks and multi-code-point emoji remain multiple tokens. Case, punctuation and whitespace are preserved. There is no normalization, BPE or subword merging. Malformed UTF-16 is rejected.

`Encode` can optionally add BOS and EOS. Unknown scalars map to UNK; `Decode` renders UNK as `�`, skips PAD/BOS, and stops at EOS by default. It validates every supplied ID, even IDs after EOS. `stopAtEos: false` skips EOS markers but continues decoding. Vocabulary snapshots are copied.

```csharp
using NeuralNetworks;

var tokenizer = CharacterTokenizer.FromCorpus(["Olá 🌍!", "ola", "oi"]);
var ids = tokenizer.Encode("Olá 🌍!", addBos: true, addEos: true);
var roundtrip = tokenizer.Decode(ids); // "Olá 🌍!"
var unknown = tokenizer.Encode("?");  // [CharacterTokenizer.UnkId]
```

### Embeddings and stacks

Source and decoder token IDs select rows in separate trainable `vocabularySize × modelSize` matrices. IDs are categorical indices, not numeric feature values; ID 8 is not treated as twice ID 4. Sinusoidal positions are added to the selected vectors. Repeated tokens select the same parameter nodes, so their gradient contributions accumulate.

The final decoder vectors are mapped to `vocabularySize` logits by a linear output head. For one valid target ID `y`:

```text
p_k = exp(logit_k - maxLogit) / Σ_j exp(logit_j - maxLogit)
CE  = log(Σ_j exp(logit_j - maxLogit)) - (logit_y - maxLogit)
loss = mean CE over valid target IDs, including EOS
```

The implementation computes log-sum-exp directly rather than `-log(p_y)`, so an extremely unlikely target does not become `log(0)` merely because its probability underflows. Loss is measured in natural-log units. This mean-token objective differs from the numerical network's half **summed** squared error.

### Teacher forcing, padding and generation

Training methods take **source IDs and target IDs**, without a caller-supplied decoder input. Targets must end with their only EOS and contain no BOS. The model automatically constructs `[BOS, target_0, …, target_(last-1)]`. Causal masks in **every decoder layer** prevent a position from reading later target inputs.

Token arrays may contain a valid prefix followed by PAD. Their supplied lengths must fit the configured capacities. PAD suffixes are trimmed before graph construction; holes, empty/all-PAD sequences and out-of-vocabulary IDs are rejected. Predictions return one row per valid decoder-prefix position; padding produces no extra rows and contributes no loss.

`Generate` begins with BOS, repeatedly chooses the largest allowed vocabulary logit, feeds that ID back, and stops at EOS or a requested cap. PAD and BOS cannot be generated; EOS, UNK and ordinary characters can. Comparing logits directly avoids underflow when excluded structural classes have very large scores. Ties select the lowest allowed ID. Returned IDs include EOS if emitted, and `EndedWithEos` distinguishes EOS termination from reaching the cap. Decoding hides structural tokens.

## How it was programmed

- [CharacterTokenizer.cs](../../src/NeuralNetworks/CharacterTokenizer.cs) validates UTF-16 with `Rune.DecodeFromUtf16`, builds sorted IDs, freezes the vocabulary and checks token boundaries.
- [EmbeddingBlock.cs](../../src/NeuralNetworks/Internal/EmbeddingBlock.cs) performs row lookup and adds positions.
- [TransformerStacks.cs](../../src/NeuralNetworks/Internal/TransformerStacks.cs) allocates independent encoder/decoder blocks, iterates every layer and preserves attention snapshots per layer. All decoder cross-attention blocks read the final encoder memory.
- [TextTransformer.cs](../../src/NeuralNetworks/TextTransformer.cs) constructs shifted targets, computes categorical loss, exposes token probabilities and performs greedy generation.
- [Categorical.cs](../../src/NeuralNetworks/Internal/Categorical.cs) implements stable probabilities and log-sum-exp cross-entropy; the scalar tape supplies reverse-mode derivatives, including `log`.
- [ParameterizedModel.cs](../../src/NeuralNetworks/ParameterizedModel.cs) owns copied parameters, seeded initialization, gradient extraction and transactional SGD. Numerical `GradientNetwork` and categorical `TextTransformer` share this machinery, but **do not share their loss contracts**. The text API does not implement `IGradientNetwork`.

Embedding matrices are stored source first, then target, each by token ID and embedding coordinate. Next come every encoder block, every decoder block, and the vocabulary head. Each encoder block stores Q/K/V/output projection, attention normalization, FFN input/output projections and FFN normalization. Each decoder block stores self Q/K/V/output and normalization, cross Q/K/V/output and normalization, then FFN input/output and normalization. Linear blocks use neuron weights followed by bias; normalization stores gamma then beta. See [the common training guide](../TRAINING.md) for initialization and array ownership.

## How to use it

The complete small example memorizes two prompt/reply pairs, using two encoder layers, two decoder layers and two attention heads. Its nine-token vocabulary includes the four structural IDs and five ordinary characters:

```csharp
using NeuralNetworks;

var tokenizer = CharacterTokenizer.FromCorpus(["a", "b", "ola", "oi"]);
var model = new TextTransformer(tokenizer.VocabularySize,
    sourceLength: 2, targetLength: 4, modelSize: 8, headCount: 2,
    feedForwardSize: 8, encoderLayerCount: 2, decoderLayerCount: 2, seed: 42);
int[][] source = [tokenizer.Encode("a", addEos: true), tokenizer.Encode("b", addEos: true)];
int[][] target = [tokenizer.Encode("ola", addEos: true), tokenizer.Encode("oi", addEos: true)];
for (var epoch = 0; epoch < 800; epoch++)
    for (var i = 0; i < source.Length; i++)
        model.TrainSample(source[i], target[i], learningRate: 0.03);

var generated = model.Generate(tokenizer.Encode("a", addEos: true));
var text = tokenizer.Decode(generated.Tokens); // "ola", with EndedWithEos == true
var probabilities = model.Predict(source[0], [CharacterTokenizer.BosId]);
var attention = model.GetAttentionWeights(source[0], [CharacterTokenizer.BosId]);
// attention.Cross[layer][head][decoderQuery][sourceKey]
```

From the repository root:

```sh
dotnet build --configuration Release
dotnet run --project examples/TextGeneration --configuration Release --no-build
# Train from scratch again, then generate for an optional one-character prompt:
dotnet run --project examples/TextGeneration --configuration Release --no-build -- b
```

The executable requires loss below one tenth of its initial value and exact greedy reproduction of both training replies with EOS termination. A validated run prints:

```text
Character vocabulary: 9; encoder layers: 2; decoder layers: 2; heads: 2
Mean token cross-entropy: 2.465243 -> 0.003124
a -> ola; ids=[8,7,4,2]; EOS=True
b -> oi; ids=[8,6,2]; EOS=True
```

This is learned training-pair memorization; the replies are not returned from a lookup table. The example does not establish open-ended text quality or generalization. Each invocation trains from scratch. An unknown custom prompt is mapped through the fixed vocabulary and has no validated output expectation.

### API and capacities

`SourceLength` and `TargetLength` are maximum **token counts**, not UTF-16 character counts. Reserve a target slot for EOS. The source example also adds EOS, but the source API does not require it. The target length equals its shifted decoder length because BOS replaces the final EOS in the training input.

`Loss`, `ComputeGradient` and `TrainSample` automatically shift targets and omit PAD suffixes. Targets must have exactly one terminal EOS and no BOS. `Predict` requires a decoder prefix starting with exactly one BOS. `Generate(source, maxNewTokens: n)` requires `1 <= n <= TargetLength`; the cap includes a possible EOS. It returns a detached `TokenGeneration` containing IDs and a termination flag, without BOS. `GetAttentionWeights` returns detached `[layer][head][query][key]` arrays for encoder self, decoder self and cross-attention.

`GetParameters`/`SetParameters` copy finite arrays. Retain the exact vocabulary mapping, dimensions, layer counts and head count with any parameter snapshot. Equal vocabulary sizes can encode entirely different characters. There is no model/tokenizer checkpoint format; retain the tokenizer or reconstruct its mapping from the same set of corpus scalars.

## Validation and limitations

Tests cover Unicode roundtrips, deterministic vocabularies, unknown/special tokens, malformed UTF-16, full numerical gradients through embeddings and two-layer stacks, nonzero gradient signal in every layer, known uniform cross-entropy/logit derivatives, extremely unlikely targets, greedy ranking underflow, EOS/cap behavior, causal attention at every decoder layer, source-memory dependence, PAD invariance, detached arrays, invalid IDs/targets/capacities, seeded updates and atomic overflow rejection. Existing numerical examples remain regression tests.

The tokenizer is character-based and corpus-fitted. The model uses explicit scalar loops and tapes, post-normalization, SGD and greedy decoding. It has no subword tokenizer, dropout, Adam, minibatches, KV cache, sampling/beam search, pretrained weights, checkpoint loader or corpus/data pipeline. More layers increase parameters, allocation and gradient instability; supported depth is not a guarantee of convergence or practical scalability. Use tiny vocabularies, short sequences and independent model instances around concurrent training.

The demonstrated width 8 and learning rate 0.03 are validated training settings on macOS and Linux, not universal convergence guarantees. A fixed seed does not make every configuration learn across runtimes.

Further reading: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) for stacked encoder/decoder architecture; [.NET Rune decoding](https://learn.microsoft.com/en-us/dotnet/api/system.text.rune.decodefromutf16?view=net-10.0) for Unicode scalar validation. The tokenizer and teaching corpus are local implementations, not the original paper's subword pipeline.

[Multi-head encoder](transformer.md) · [numerical encoder-decoder](transformer-seq2seq.md) · [documentation index](../README.md).
