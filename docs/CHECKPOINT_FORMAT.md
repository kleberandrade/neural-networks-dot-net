# Text checkpoint formats v1 and v2

`TextCheckpoint` writes v2 and reads v1/v2. It stores `TextTransformer`, optional Adam state and its exact `CharacterTokenizer` mapping in one UTF-8 JSON file. It does not serialize executable code, arbitrary CLR types or the source corpus. The format has four envelope fields:

- `format`: exactly `neural-networks-dot-net.text`.
- `version`: integer `2` for new saves; `1` remains readable as the legacy SGD schema.
- `payload`: the object described below.
- `sha256`: 64 hexadecimal characters, SHA-256 of the payload's exact UTF-8 JSON text.

`Save` writes compact JSON without a BOM. Formatting or editing inside `payload` changes its checksum; use the API instead of hand-editing files. The checksum is an integrity check, not a signature. Unknown versions, unknown/missing/duplicate fields and malformed JSON fail with `InvalidDataException`. File access errors propagate as I/O exceptions.

## Payload and version compatibility

- `configuration`: `vocabularySize`, `sourceLength`, `targetLength`, `modelSize`, `headCount`, `feedForwardSize`, `encoderLayerCount`, `decoderLayerCount`, `seed`.
- `vocabulary`: the full ID-indexed string array. Entries 0–3 are `<pad>`, `<bos>`, `<eos>`, `<unk>`; later entries are unique, ascending Unicode scalars. Both versions use this tokenizer algorithm/mapping. The restored tokenizer is immutable.
- `parameters`: finite IEEE-754 doubles, in the [documented model order](architectures/transformer-text.md#how-it-was-programmed). JSON numbers roundtrip double values; no float32 quantization occurs.
- `training`: null or progress metadata. V1 has `epochsCompleted`, `shuffleSeed`, `learningRate`, `corpusFingerprint`. V2 requires these plus `batchSize` and `optimizer` (`sgd` or `adam`). SGD requires batch size 1.
- `optimizer` (**v2 only**): null for SGD/no optimizer, or Adam state with `options` (`beta1`, `beta2`, `epsilon`), `step` (nonnegative signed 64-bit integer), `firstMoment` and `secondMoment` (full-precision arrays, each exactly the model parameter count).

V1 has exactly four payload fields and no optimizer state; it restores SGD/batch size 1 without inventing Adam moments. V2 has exactly five payload fields. Adam progress requires Adam state; SGD progress requires null optimizer state. An inference-only v2 save may have null progress with optional Adam state, but CLI Adam resume requires progress to recover ordering/batch size. `Load` exposes a restored model-bound `AdamOptimizer` through `LoadedTextCheckpoint.Optimizer`, or null. Never resume Adam by constructing a fresh optimizer from loaded weights alone.

Moment values must be finite, second moments nonnegative, beta values finite in `[0,1)` and epsilon positive/finite. Step zero requires zero moments; a step at `long.MaxValue` can be loaded for inspection, but its next update rejects counter overflow atomically. Unknown/missing/duplicate state fields are errors. Array lengths are checked before deserializing moment arrays and before model construction; the overall file byte limit covers all three parameter-sized arrays. Adam checkpoints are correspondingly larger than SGD files.

Dimensions are positive, model width is at least two and divisible by head count, both stacks have positive depth, and vocabulary includes the four fixed IDs plus at least one scalar. Input and target lengths count tokens including any explicit EOS. The parameter count must agree exactly with the architecture before allocation.

For width `d`, vocabulary `v`, FFN width `f`, encoder count `E` and decoder count `D`:

```text
Linear(in, out) = out × (in + 1)        includes bias
Attention      = 4 × Linear(d, d)      Q, K, V, output
Normalization  = 2 × d                gamma, beta
FFN            = Linear(d, f) + Linear(f, d) + Normalization
EncoderBlock   = Attention + Normalization + FFN
DecoderBlock   = 2×Attention + 2×Normalization + FFN
Total          = 2×v×d + E×EncoderBlock + D×DecoderBlock + Linear(d, v)
```

Load checks count arithmetic and default resource bounds before constructing a model: 32 MiB file, one million parameters, 65,536 vocabulary IDs, 4,096 source/target tokens and 64 layers per stack. Callers can pass `TextCheckpointLimits` to change these explicit bounds. The scalar implementation remains expensive even below them.

The fixed v1/v2 architecture includes separate source/target embeddings with sinusoidal positions, post-normalization with epsilon `1e-5`, ReLU FFNs and a vocabulary head trained with mean categorical cross-entropy. Changes to parameter layout or tokenizer semantics require a new format version; do not silently reinterpret existing arrays. V2 extends training state without changing the model parameter order.

## Save, reload and resume

`TextCheckpoint.Save(path, model, tokenizer, training, optimizer)` validates supplied objects, snapshots the parameters, writes and flushes a unique sibling temporary file, then replaces the destination through a same-directory rename. The temporary file is removed on failure. Invalid save arguments leave an existing checkpoint unchanged. An existing destination is intentionally replaceable through this API; the training CLI additionally refuses accidental new-training overwrite unless requested.

`Load` validates the envelope/checksum, configuration, count/limits, vocabulary and finite values, constructs the exact model, then restores parameters. Its `LoadedTextCheckpoint` exposes `Model`, `Tokenizer` and optional `Training` and model-bound `Optimizer`. It never falls back to random weights after a failed load.

The initial seed is provenance; restored weights, not a new random initialization, determine inference. Sampling seeds are chosen per inference call. Training progress describes **completed epochs**. The corpus fingerprint hashes canonical parsed prompt/response pairs in order; JSONL whitespace/property order and filename changes do not alter it, while changed sample strings/order do. Plain-text window parameters affect those parsed pairs.

SGD has no moment buffers. Adam must also restore both moment arrays, beta values, epsilon and the step counter. Matching weights, optimizer state, dataset/order, epoch number, batch size, rate and shuffle seed reproduce continued steps on the same runtime. A fixed seed does not imply bit-identical optimization across platforms or future runtimes. Saving with the wrong historically associated tokenizer of equal size cannot be detected from weights alone; the API preserves the exact pair supplied, and the CLI always uses the fitted/loaded pair.

The original corpus is unnecessary for inference after loading. Keep it for repeatable training/evaluation. The format is a teaching checkpoint, not a generic ONNX/safetensors importer, generic optimizer archive or guaranteed compatibility layer for future architecture revisions.

[Corpus/checkpoint walkthrough](architectures/transformer-corpus-checkpoints.md) · [text model](architectures/transformer-text.md).

[Minibatch Adam walkthrough](architectures/transformer-minibatch-adam.md) includes equations, the executable resume demo and library usage.
