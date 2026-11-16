using NeuralNetworks.Internal;

namespace NeuralNetworks;

/// <summary>Detached attention matrices, indexed by head, query position, then key position.</summary>
public sealed record TransformerAttentionWeights(double[][][] EncoderSelf, double[][][] DecoderSelf, double[][][] Cross);

/// <summary>One multi-head encoder and one causal decoder for fixed-length numerical sequences.</summary>
public sealed class TransformerSequenceToSequence : GradientNetwork
{
    private readonly LinearBlock sourceProjection;
    private readonly LinearBlock targetProjection;
    private readonly LinearBlock output;
    private readonly TransformerEncoderLayer encoder;
    private readonly MultiHeadAttention decoderSelf;
    private readonly MultiHeadAttention cross;
    private readonly LayerNormalization selfNorm;
    private readonly LayerNormalization crossNorm;
    private readonly TransformerFeedForward decoderFeedForward;
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

    public TransformerSequenceToSequence(int sourceLength, int targetLength, int inputSize, int outputSize,
        int modelSize = 4, int headCount = 2, int feedForwardSize = 8, int seed = 0)
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
        noSourcePadding = new bool[sourceLength];
        noTargetPadding = new bool[targetLength];
        sourceProjection = AddLinear(inputSize, modelSize);
        targetProjection = AddLinear(outputSize, modelSize);
        var encoderAttention = AddAttention();
        var encoderNorm = AddNormalization(modelSize);
        encoder = new TransformerEncoderLayer(encoderAttention, encoderNorm, AddFeedForward(feedForwardSize));
        decoderSelf = AddAttention();
        selfNorm = AddNormalization(modelSize);
        cross = AddAttention();
        crossNorm = AddNormalization(modelSize);
        decoderFeedForward = AddFeedForward(feedForwardSize);
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
        bool[]? sourcePaddingMask = null, bool[]? targetPaddingMask = null)
    {
        var sample = Prepare(source, decoderInput, sourcePaddingMask, targetPaddingMask);
        var graph = CreateGraph(sample.Input);
        var result = Build(graph.Input, graph.Parameters, sample.SourcePadding, sample.TargetPadding);
        return new TransformerAttentionWeights(TransformerLayers.Snapshot(result.Encoder),
            TransformerLayers.Snapshot(result.Decoder), TransformerLayers.Snapshot(result.Cross));
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
        var memory = encoder.Apply(source, parameters, sourcePadding);
        var target = TransformerLayers.Embed(input.AsSpan(SourceSize).ToArray(), parameters,
            targetProjection, TargetLength, OutputFeatures, ModelSize, targetPadding);
        var selfAttention = decoderSelf.Apply(target, target, parameters, targetPadding, causal: true);
        var first = TransformerLayers.Residual(target, selfAttention.Output, parameters, selfNorm);
        var crossAttention = cross.Apply(first, memory.Output, parameters, sourcePadding);
        var second = TransformerLayers.Residual(first, crossAttention.Output, parameters, crossNorm);
        var decoded = decoderFeedForward.Apply(second, parameters);
        var result = decoded.SelectMany((token, t) => targetPadding[t]
            ? Operations.Zeros(input[0], OutputFeatures)
            : Operations.Sigmoid(output.Apply(parameters, token))).ToArray();
        return new Evaluation(result, memory.Weights, selfAttention.Weights, crossAttention.Weights);
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

    private MultiHeadAttention AddAttention() => new(AddLinear(ModelSize, ModelSize), AddLinear(ModelSize, ModelSize),
        AddLinear(ModelSize, ModelSize), AddLinear(ModelSize, ModelSize), ModelSize, HeadCount);

    private TransformerFeedForward AddFeedForward(int size) =>
        new(AddLinear(ModelSize, size), AddLinear(size, ModelSize), AddNormalization(ModelSize));

    private static int Count(int length, int features) =>
        checked(Guard.Positive(length, nameof(length)) * Guard.Positive(features, nameof(features)));

    private readonly record struct Prepared(double[] Input, bool[] SourcePadding, bool[] TargetPadding, bool[] OutputMask);
    private readonly record struct Evaluation(Scalar[] Output, Scalar[][][] Encoder, Scalar[][][] Decoder, Scalar[][][] Cross);
}
