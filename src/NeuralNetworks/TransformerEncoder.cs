using NeuralNetworks.Internal;

namespace NeuralNetworks;

/// <summary>One bidirectional encoder block: single-head attention, residuals, layer norms and ReLU FFN.</summary>
public sealed class TransformerEncoder : GradientNetwork
{
    private readonly LinearBlock projection;
    private readonly LinearBlock query;
    private readonly LinearBlock key;
    private readonly LinearBlock value;
    private readonly LinearBlock attentionOutput;
    private readonly LinearBlock feedForwardIn;
    private readonly LinearBlock feedForwardOut;
    private readonly LinearBlock output;
    private readonly LayerNormalization norm1;
    private readonly LayerNormalization norm2;
    public int SequenceLength { get; }
    public int Features { get; }
    public int ModelSize { get; }

    public TransformerEncoder(int sequenceLength, int inputSize, int modelSize, int outputSize, int feedForwardSize = 4, int seed = 0)
        : base(checked(Guard.Positive(sequenceLength, nameof(sequenceLength)) * Guard.Positive(inputSize, nameof(inputSize))), outputSize, seed)
    {
        if (modelSize < 2) throw new ArgumentOutOfRangeException(nameof(modelSize), "Use at least two model dimensions.");
        Guard.Positive(feedForwardSize, nameof(feedForwardSize));
        SequenceLength = sequenceLength;
        Features = inputSize;
        ModelSize = modelSize;
        projection = AddLinear(inputSize, modelSize);
        query = AddLinear(modelSize, modelSize);
        key = AddLinear(modelSize, modelSize);
        value = AddLinear(modelSize, modelSize);
        attentionOutput = AddLinear(modelSize, modelSize);
        feedForwardIn = AddLinear(modelSize, feedForwardSize);
        feedForwardOut = AddLinear(feedForwardSize, modelSize);
        output = AddLinear(modelSize, outputSize);
        double[] normalization = [.. Enumerable.Repeat(1.0, modelSize), .. new double[modelSize]];
        norm1 = new LayerNormalization(modelSize, AddParameters(normalization));
        norm2 = new LayerNormalization(modelSize, AddParameters(normalization));
    }

    /// <summary>Returns detached attention rows indexed by query position, then key position.</summary>
    public double[][] GetAttentionWeights(double[] input)
    {
        var graph = CreateGraph(input);
        var tokens = Embed(graph.Input, graph.Parameters);
        var keys = tokens.Select(t => key.Apply(graph.Parameters, t)).ToArray();
        return tokens.Select(t => AttentionWeights(query.Apply(graph.Parameters, t), keys)
            .Select(x => x.Value).ToArray()).ToArray();
    }

    private protected override Scalar[] Forward(Scalar[] input, Scalar[] parameters)
    {
        var tokens = Embed(input, parameters);
        var queries = tokens.Select(t => query.Apply(parameters, t)).ToArray();
        var keys = tokens.Select(t => key.Apply(parameters, t)).ToArray();
        var values = tokens.Select(t => value.Apply(parameters, t)).ToArray();
        var pooled = Operations.Zeros(input[0], ModelSize);
        for (var t = 0; t < SequenceLength; t++)
        {
            var context = Attend(queries[t], keys, values);
            var attended = attentionOutput.Apply(parameters, context);
            var residual = new Scalar[ModelSize];
            for (var i = 0; i < ModelSize; i++) residual[i] = tokens[t][i] + attended[i];
            var first = norm1.Apply(parameters, residual);
            var feedForward = feedForwardOut.Apply(parameters, feedForwardIn.Apply(parameters, first).Select(x => x.Relu()).ToArray());
            for (var i = 0; i < ModelSize; i++) residual[i] = first[i] + feedForward[i];
            var encoded = norm2.Apply(parameters, residual);
            for (var i = 0; i < ModelSize; i++) pooled[i] += encoded[i] / SequenceLength;
        }
        return Operations.Sigmoid(output.Apply(parameters, pooled));
    }

    private Scalar[][] Embed(Scalar[] input, Scalar[] parameters)
    {
        var tokens = new Scalar[SequenceLength][];
        for (var t = 0; t < SequenceLength; t++)
        {
            tokens[t] = projection.Apply(parameters, input.AsSpan(t * Features, Features).ToArray());
            for (var i = 0; i < ModelSize; i++)
            {
                var angle = t / Math.Pow(10000, 2.0 * (i / 2) / ModelSize);
                tokens[t][i] += i % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle);
            }
        }
        return tokens;
    }

    private Scalar[] Attend(Scalar[] queryToken, Scalar[][] keys, Scalar[][] values)
    {
        var weights = AttentionWeights(queryToken, keys);
        var context = Operations.Zeros(queryToken[0], ModelSize);
        for (var t = 0; t < SequenceLength; t++)
            for (var i = 0; i < ModelSize; i++)
                context[i] += weights[t] * values[t][i];
        return context;
    }

    private Scalar[] AttentionWeights(Scalar[] queryToken, Scalar[][] keys)
    {
        var scores = keys.Select(k => Operations.Dot(queryToken, k) / Math.Sqrt(ModelSize)).ToArray();
        var shift = scores.Max(s => s.Value);
        var exponentials = scores.Select(s => (s + (-shift)).Exp()).ToArray();
        var inverseSum = Operations.Sum(exponentials).Reciprocal();
        return exponentials.Select(x => x * inverseSum).ToArray();
    }
}
