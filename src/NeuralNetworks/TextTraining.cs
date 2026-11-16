namespace NeuralNetworks;

/// <summary>Validated SGD epochs and mean-gradient Adam minibatches.</summary>
public static class TextTraining
{
    public static double TrainEpoch(TextTransformer model, IReadOnlyList<EncodedTextPair> samples, double learningRate, int shuffleSeed, int epoch)
    {
        ArgumentNullException.ThrowIfNull(model); ArgumentNullException.ThrowIfNull(samples);
        Guard.LearningRate(learningRate);
        if (epoch < 0) throw new ArgumentOutOfRangeException(nameof(epoch));
        if (samples.Count == 0) throw new ArgumentException("Provide at least one training sample.", nameof(samples));
        var copied = CopySamples(model, samples);
        var order = Enumerable.Range(0, copied.Length).ToArray();
        new Random(unchecked(shuffleSeed + epoch)).Shuffle(order);
        var loss = 0.0;
        foreach (var index in order) loss += model.TrainSample(copied[index].Source, copied[index].Target, learningRate);
        return loss / copied.Length;
    }

    /// <summary>Equal mean of per-sample mean token losses/gradients, followed by one Adam step.</summary>
    public static double TrainBatch(TextTransformer model, IReadOnlyList<EncodedTextPair> samples, AdamOptimizer optimizer, double learningRate)
    {
        ValidateAdam(model, optimizer, learningRate);
        return TrainCopiedBatch(model, CopySamples(model, samples), optimizer, learningRate);
    }

    public static double TrainEpoch(TextTransformer model, IReadOnlyList<EncodedTextPair> samples, AdamOptimizer optimizer,
        int batchSize, double learningRate, int shuffleSeed, int epoch)
    {
        ValidateAdam(model, optimizer, learningRate); Guard.Positive(batchSize, nameof(batchSize));
        if (epoch < 0) throw new ArgumentOutOfRangeException(nameof(epoch));
        var copied = CopySamples(model, samples); var order = Enumerable.Range(0, copied.Length).ToArray();
        new Random(unchecked(shuffleSeed + epoch)).Shuffle(order);
        var loss = 0.0;
        foreach (var indices in order.Chunk(batchSize))
        {
            var batch = indices.Select(i => copied[i]).ToArray();
            loss = Guard.Finite(loss + TrainCopiedBatch(model, batch, optimizer, learningRate) * (batch.Length / (double)copied.Length));
        }
        return loss;
    }

    private static double TrainCopiedBatch(TextTransformer model, EncodedTextPair[] samples, AdamOptimizer optimizer, double rate)
    {
        var gradient = new double[model.ParameterCount]; var loss = 0.0;
        foreach (var sample in samples)
        {
            var result = model.LossAndGradient(sample.Source, sample.Target);
            loss = Guard.Finite(loss + result.Loss / samples.Length);
            for (var i = 0; i < gradient.Length; i++) gradient[i] = Guard.Finite(gradient[i] + result.Gradient[i] / samples.Length);
        }
        optimizer.Step(gradient, rate);
        return loss;
    }

    private static void ValidateAdam(TextTransformer model, AdamOptimizer optimizer, double rate)
    {
        ArgumentNullException.ThrowIfNull(model); ArgumentNullException.ThrowIfNull(optimizer);
        Guard.LearningRate(rate); optimizer.EnsureModel(model);
    }

    private static EncodedTextPair[] CopySamples(TextTransformer model, IReadOnlyList<EncodedTextPair> samples)
    {
        ArgumentNullException.ThrowIfNull(samples);
        if (samples.Count == 0) throw new ArgumentException("Provide at least one training sample.", nameof(samples));
        return samples.Select(sample =>
        {
            ArgumentNullException.ThrowIfNull(sample);
            return model.CopyTrainingPair(sample.Source, sample.Target);
        }).ToArray();
    }
}
