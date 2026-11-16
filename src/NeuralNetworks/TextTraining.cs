namespace NeuralNetworks;

/// <summary>One validated, deterministic SGD epoch; no optimizer state beyond model parameters.</summary>
public static class TextTraining
{
    public static double TrainEpoch(TextTransformer model, IReadOnlyList<EncodedTextPair> samples, double learningRate, int shuffleSeed, int epoch)
    {
        ArgumentNullException.ThrowIfNull(model); ArgumentNullException.ThrowIfNull(samples);
        Guard.LearningRate(learningRate);
        if (epoch < 0) throw new ArgumentOutOfRangeException(nameof(epoch));
        if (samples.Count == 0) throw new ArgumentException("Provide at least one training sample.", nameof(samples));
        var copied = samples.Select(sample =>
        {
            ArgumentNullException.ThrowIfNull(sample);
            return model.CopyTrainingPair(sample.Source, sample.Target);
        }).ToArray();
        var order = Enumerable.Range(0, copied.Length).ToArray();
        new Random(unchecked(shuffleSeed + epoch)).Shuffle(order);
        var loss = 0.0;
        foreach (var index in order) loss += model.TrainSample(copied[index].Source, copied[index].Target, learningRate);
        return loss / copied.Length;
    }
}
