using NeuralNetworks.Internal;

namespace NeuralNetworks;

/// <summary>Detached attention matrices, indexed by head, query position, then key position.</summary>
public sealed record TransformerAttentionWeights(double[][][] EncoderSelf, double[][][] DecoderSelf, double[][][] Cross);

/// <summary>Detached per-layer attention: layer, head, query, key.</summary>
public sealed record TransformerStackAttentionWeights(double[][][][] EncoderSelf, double[][][][] DecoderSelf, double[][][][] Cross);

/// <summary>Stacks of multi-head encoders and causal decoders for fixed-length numerical sequences.</summary>
public sealed class TransformerSequenceToSequence : GradientNetwork
{
    private readonly LinearBlock sourceProjection;
    private readonly LinearBlock targetProjection;
    private readonly LinearBlock output;
    private readonly TransformerEncoderLayer[] encoders;
    private readonly TransformerDecoderLayer[] decoders;
    private readonly bool[] noSourcePadding;
    private readonly bool[] noTargetPadding;
    public int SourceLength { get; }
    public int TargetLength { get; }
    public int InputFeatures { get; }
    public int OutputFeatures { get; }
    public int SourceSize { get; }
    public int TargetSize { get; }
    public int ModelSize { get; }
    public int HeadCount { get; }
    public int EncoderLayerCount { get; }
    public int DecoderLayerCount { get; }

    public TransformerSequenceToSequence(int sourceLength, int targetLength, int inputSize, int outputSize,
        int modelSize = 4, int headCount = 2, int feedForwardSize = 8, int seed = 0, int encoderLayerCount = 1, int decoderLayerCount = 1)
        : base(checked(Count(sourceLength, inputSize) + Count(targetLength, outputSize)), Count(targetLength, outputSize), seed)
    {
        MultiHeadAttention.Validate(modelSize, headCount);
        Guard.Positive(feedForwardSize, nameof(feedForwardSize));
        SourceLength = sourceLength;
        TargetLength = targetLength;
        InputFeatures = inputSize;
        OutputFeatures = outputSize;
        SourceSize = Count(sourceLength, inputSize);
        TargetSize = Count(targetLength, outputSize);
        ModelSize = modelSize;
        HeadCount = headCount;
        EncoderLayerCount = Guard.Positive(encoderLayerCount, nameof(encoderLayerCount));
        DecoderLayerCount = Guard.Positive(decoderLayerCount, nameof(decoderLayerCount));
        noSourcePadding = new bool[sourceLength];
        noTargetPadding = new bool[targetLength];
        sourceProjection = AddLinear(inputSize, modelSize);
        targetProjection = AddLinear(outputSize, modelSize);
        var builder = new TransformerBuilder(AddLinear, AddNormalization, modelSize, headCount, feedForwardSize);
        encoders = Enumerable.Range(0, EncoderLayerCount).Select(_ => builder.Encoder()).ToArray();
        decoders = Enumerable.Range(0, DecoderLayerCount).Select(_ => builder.Decoder()).ToArray();
        output = AddLinear(modelSize, outputSize);
    }

    public double[] Predict(double[] source, double[] decoderInput, bool[]? sourcePaddingMask = null, bool[]? targetPaddingMask = null)
    {
        var sample = Prepare(source, decoderInput, sourcePaddingMask, targetPaddingMask);
        return PredictUsing(sample.Input, (x, p) => Build(x, p, sample.SourcePadding, sample.TargetPadding).Output);
    }

    public double Loss(double[] source, double[] decoderInput, double[] target, bool[]? sourcePaddingMask = null, bool[]? targetPaddingMask = null)
    {
        var sample = Prepare(source, decoderInput, sourcePaddingMask, targetPaddingMask);
        return LossUsing(sample.Input, target, (x, p) => Build(x, p, sample.SourcePadding, sample.TargetPadding).Output, sample.OutputMask);
    }

    public double[] ComputeGradient(double[] source, double[] decoderInput, double[] target, bool[]? sourcePaddingMask = null, bool[]? targetPaddingMask = null)
    {
        var sample = Prepare(source, decoderInput, sourcePaddingMask, targetPaddingMask);
        return GradientUsing(sample.Input, target, (x, p) => Build(x, p, sample.SourcePadding, sample.TargetPadding).Output, sample.OutputMask);
    }

    public double TrainSample(double[] source, double[] decoderInput, double[] target, double learningRate,
        bool[]? sourcePaddingMask = null, bool[]? targetPaddingMask = null)
    {
        var sample = Prepare(source, decoderInput, sourcePaddingMask, targetPaddingMask);
        return TrainUsing(sample.Input, target, learningRate,
            (x, p) => Build(x, p, sample.SourcePadding, sample.TargetPadding).Output, sample.OutputMask);
    }

    public TransformerAttentionWeights GetAttentionWeights(double[] source, double[] decoderInput,
        bool[]? sourcePaddingMask = null, bool[]? targetPaddingMask = null, int encoderLayerIndex = 0, int decoderLayerIndex = 0)
    {
        if (encoderLayerIndex < 0 || encoderLayerIndex >= EncoderLayerCount) throw new ArgumentOutOfRangeException(nameof(encoderLayerIndex));
        if (decoderLayerIndex < 0 || decoderLayerIndex >= DecoderLayerCount) throw new ArgumentOutOfRangeException(nameof(decoderLayerIndex));
        var layers = GetLayerAttentionWeights(source, decoderInput, sourcePaddingMask, targetPaddingMask);
        return new TransformerAttentionWeights(layers.EncoderSelf[encoderLayerIndex], layers.DecoderSelf[decoderLayerIndex], layers.Cross[decoderLayerIndex]);
    }

    public TransformerStackAttentionWeights GetLayerAttentionWeights(double[] source, double[] decoderInput,
        bool[]? sourcePaddingMask = null, bool[]? targetPaddingMask = null)
    {
        var sample = Prepare(source, decoderInput, sourcePaddingMask, targetPaddingMask);
        var graph = CreateGraph(sample.Input);
        var result = Build(graph.Input, graph.Parameters, sample.SourcePadding, sample.TargetPadding);
        return new TransformerStackAttentionWeights(TransformerStacks.Snapshot(result.Encoder),
            TransformerStacks.Snapshot(result.Decoder), TransformerStacks.Snapshot(result.Cross));
    }

    /// <summary>Autoregressively feeds predictions back into the decoder; never takes true target values.</summary>
    public double[] Generate(double[] source, double[] startToken, bool[]? sourcePaddingMask = null)
    {
        Guard.Vector(source, SourceSize, nameof(source));
        Guard.Vector(startToken, OutputFeatures, nameof(startToken));
        var sourceMask = TransformerLayers.Padding(sourcePaddingMask, SourceLength, nameof(sourcePaddingMask));
        var decoder = new double[TargetSize];
        var generated = new double[TargetSize];
        startToken.CopyTo(decoder, 0);
        for (var step = 0; step < TargetLength; step++)
        {
            var prediction = Predict(source, decoder, sourceMask);
            Array.Copy(prediction, step * OutputFeatures, generated, step * OutputFeatures, OutputFeatures);
            if (step + 1 < TargetLength)
                Array.Copy(prediction, step * OutputFeatures, decoder, (step + 1) * OutputFeatures, OutputFeatures);
        }
        return generated;
    }

    private protected override Scalar[] Forward(Scalar[] input, Scalar[] parameters) =>
        Build(input, parameters, noSourcePadding, noTargetPadding).Output;

    private Evaluation Build(Scalar[] input, Scalar[] parameters, bool[] sourcePadding, bool[] targetPadding)
    {
        var source = TransformerLayers.Embed(input.AsSpan(0, SourceSize).ToArray(), parameters,
            sourceProjection, SourceLength, InputFeatures, ModelSize, sourcePadding);

        var target = TransformerLayers.Embed(input.AsSpan(SourceSize).ToArray(), parameters,
            targetProjection, TargetLength, OutputFeatures, ModelSize, targetPadding);
        var stack = TransformerStacks.Apply(source, target, parameters, encoders, decoders, sourcePadding, targetPadding);
        var decoded = stack.Output;
        var result = decoded.SelectMany((token, t) => targetPadding[t]
            ? Operations.Zeros(input[0], OutputFeatures)
            : Operations.Sigmoid(output.Apply(parameters, token))).ToArray();
        return new Evaluation(result, stack.Encoder, stack.Decoder, stack.Cross);
    }

    private Prepared Prepare(double[] source, double[] decoderInput, bool[]? sourcePaddingMask, bool[]? targetPaddingMask)
    {
        Guard.Vector(source, SourceSize, nameof(source));
        Guard.Vector(decoderInput, TargetSize, nameof(decoderInput));
        var sourceMask = TransformerLayers.Padding(sourcePaddingMask, SourceLength, nameof(sourcePaddingMask));
        var targetMask = TransformerLayers.Padding(targetPaddingMask, TargetLength, nameof(targetPaddingMask));
        return new Prepared([.. source, .. decoderInput], sourceMask, targetMask,
            targetMask.SelectMany(pad => Enumerable.Repeat(pad, OutputFeatures)).ToArray());
    }

    private static int Count(int length, int features) =>
        checked(Guard.Positive(length, nameof(length)) * Guard.Positive(features, nameof(features)));

    private readonly record struct Prepared(double[] Input, bool[] SourcePadding, bool[] TargetPadding, bool[] OutputMask);
    private readonly record struct Evaluation(Scalar[] Output, Scalar[][][][] Encoder, Scalar[][][][] Decoder, Scalar[][][][] Cross);
}
