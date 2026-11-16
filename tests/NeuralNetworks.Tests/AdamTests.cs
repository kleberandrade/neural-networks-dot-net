using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using NeuralNetworks;

internal static class AdamTests
{
    private static TextTransformer Model(int seed = 7) => new(7, 3, 4, 4, 2, 3, 1, 1, seed);
    private static readonly EncodedTextPair[] Samples = [new([4, 2], [5, 2]), new([5, 4, 2], [6, 5, 2]), new([6, 2], [4, 2])];
    private static void SameState(AdamState a, AdamState b)
    {
        Check.True(a.Step == b.Step && a.Options == b.Options, "Adam counter/options changed.");
        Check.Same(a.FirstMoment, b.FirstMoment); Check.Same(a.SecondMoment, b.SecondMoment);
    }

    public static void KnownUpdates()
    {
        var model = Model(); var original = model.GetParameters();
        var optimizer = new AdamOptimizer(model, new AdamOptions(0.8, 0.9, 1e-6));
        var gradient = new double[model.ParameterCount]; gradient[0] = 2; gradient[^1] = -3;
        optimizer.Step(gradient, 0.01);
        var state = optimizer.GetState();
        Check.Near(0.4, state.FirstMoment[0]); Check.Near(0.4, state.SecondMoment[0]);
        Check.Near(original[0] - 0.01 * 2 / (2 + 1e-6), model.GetParameters()[0]);
        gradient[0] = -1; gradient[^1] = 0;
        optimizer.Step(gradient, 0.01); state = optimizer.GetState();
        Check.Near(0.12, state.FirstMoment[0]); Check.Near(0.46, state.SecondMoment[0]);
        var expected = original[0] - 0.01 * 2 / (2 + 1e-6) - 0.01 * (0.12 / 0.36) / (Math.Sqrt(0.46 / 0.19) + 1e-6);
        Check.Near(expected, model.GetParameters()[0]); Check.True(state.Step == 2, "Adam step count differs.");
        state.FirstMoment[0] = 99; Check.Near(0.12, optimizer.GetState().FirstMoment[0]);
    }

    public static void MinibatchMean()
    {
        var model = Model(); var expected = Model();
        var gradients = Samples.Take(2).Select(s => model.ComputeGradient(s.Source, s.Target)).ToArray();
        var mean = Enumerable.Range(0, model.ParameterCount).Select(i => gradients.Sum(g => g[i] / 2)).ToArray();
        var loss = Samples.Take(2).Average(s => model.Loss(s.Source, s.Target));
        var expectedAdam = new AdamOptimizer(expected); expectedAdam.Step(mean, 0.003);
        var optimizer = new AdamOptimizer(model);
        Check.Near(loss, TextTraining.TrainBatch(model, Samples.Take(2).ToArray(), optimizer, 0.003));
        Check.Same(expected.GetParameters(), model.GetParameters()); SameState(expectedAdam.GetState(), optimizer.GetState());
        Check.True(optimizer.GetState().Step == 1, "Batch made multiple updates.");
        // Three records with batch size two must keep a final batch of one.
        var manual = Model(); var manualAdam = new AdamOptimizer(manual); var order = Enumerable.Range(0, Samples.Length).ToArray();
        new Random(42).Shuffle(order);
        foreach (var batch in order.Chunk(2)) TextTraining.TrainBatch(manual, batch.Select(i => Samples[i]).ToArray(), manualAdam, 0.003);
        var epochModel = Model(); var epochAdam = new AdamOptimizer(epochModel);
        TextTraining.TrainEpoch(epochModel, Samples, epochAdam, 2, 0.003, 42, 0);
        Check.Same(manual.GetParameters(), epochModel.GetParameters()); SameState(manualAdam.GetState(), epochAdam.GetState());
        Check.True(epochAdam.GetState().Step == 2, "Partial minibatch lost or padded.");
    }

    public static void Boundaries()
    {
        var model = Model(); var optimizer = new AdamOptimizer(model); var before = model.GetParameters(); var state = optimizer.GetState();
        Check.Throws<ArgumentOutOfRangeException>(() => new AdamOptimizer(model, new AdamOptions(Beta1: 1)));
        Check.Throws<ArgumentOutOfRangeException>(() => new AdamOptimizer(model, new AdamOptions(Epsilon: double.NaN)));
        Check.Throws<ArgumentException>(() => optimizer.Step([1], 0.01));
        var bad = new double[model.ParameterCount]; bad[^1] = double.NaN;
        Check.Throws<ArgumentException>(() => optimizer.Step(bad, 0.01));
        bad[^1] = double.MaxValue;
        Check.Throws<ArithmeticException>(() => optimizer.Step(bad, 0.01));
        Check.Same(before, model.GetParameters()); SameState(state, optimizer.GetState());
        Check.Throws<ArgumentException>(() => TextTraining.TrainBatch(model, Samples, new AdamOptimizer(Model()), 0.01));
        Check.Throws<ArgumentException>(() => TextTraining.TrainBatch(model, [Samples[0], new([4], [4])], optimizer, 0.01));
        Check.Throws<ArgumentException>(() => TextTraining.TrainEpoch(model, [Samples[0], new([4], [4])], optimizer, 1, 0.01, 42, 0));
        Check.Throws<ArgumentOutOfRangeException>(() => TextTraining.TrainEpoch(model, Samples, optimizer, 0, 0.01, 42, 0));
        Check.Throws<ArgumentException>(() => TextTraining.TrainBatch(model, [], optimizer, 0.01));
        Check.Throws<ArgumentNullException>(() => TextTraining.TrainBatch(model, Samples, null!, 0.01));
        Check.Throws<ArgumentOutOfRangeException>(() => optimizer.Step(new double[model.ParameterCount], 0));
        Check.Same(before, model.GetParameters()); SameState(state, optimizer.GetState());
        var invalid = state with { SecondMoment = Enumerable.Repeat(-1.0, model.ParameterCount).ToArray() };
        Check.Throws<ArgumentException>(() => AdamOptimizer.Restore(model, invalid));
        var copy = state with { FirstMoment = (double[])state.FirstMoment.Clone() };
        var restored = AdamOptimizer.Restore(model, copy); copy.FirstMoment[0] = 99;
        Check.Near(0, restored.GetState().FirstMoment[0]);
        var extremeState = state with { Step = 1, SecondMoment = Enumerable.Repeat(double.MaxValue, model.ParameterCount).ToArray() };
        var extreme = AdamOptimizer.Restore(model, extremeState);
        Check.Throws<ArithmeticException>(() => extreme.Step(new double[model.ParameterCount], 0.01));
        Check.Same(before, model.GetParameters()); SameState(extremeState, extreme.GetState());
        var maximumStep = AdamOptimizer.Restore(model, state with { Step = long.MaxValue });
        Check.Throws<OverflowException>(() => maximumStep.Step(new double[model.ParameterCount], 0.01));
        Check.Same(before, model.GetParameters()); Check.True(maximumStep.GetState().Step == long.MaxValue, "Rejected step changed counter.");
    }

    public static void CheckpointResume()
    {
        var root = Path.Combine(Path.GetTempPath(), "adam-tests-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(root);
        try
        {
            var a = Model(); var adamA = new AdamOptimizer(a, new AdamOptions(0.85, 0.95, 1e-7));
            var b = Model(); var adamB = new AdamOptimizer(b, new AdamOptions(0.85, 0.95, 1e-7));
            TextTraining.TrainEpoch(a, Samples, adamA, 2, 0.003, 42, 0);
            TextTraining.TrainEpoch(a, Samples, adamA, 2, 0.003, 42, 1);
            TextTraining.TrainEpoch(b, Samples, adamB, 2, 0.003, 42, 0);
            var tokenizer = CharacterTokenizer.FromCorpus(["abc"]); var path = Path.Combine(root, "adam.json");
            var progress = new TextTrainingState(1, 42, 0.003, new string('a', 64), 2, "adam");
            TextCheckpoint.Save(path, b, tokenizer, progress, adamB);
            var loaded = TextCheckpoint.Load(path);
            Check.True(loaded.Training == progress && loaded.Optimizer is not null, "Optimizer/progress not restored.");
            SameState(adamB.GetState(), loaded.Optimizer!.GetState());
            TextTraining.TrainEpoch(loaded.Model, Samples, loaded.Optimizer, 2, 0.003, 42, loaded.Training!.EpochsCompleted);
            Check.Same(a.GetParameters(), loaded.Model.GetParameters()); SameState(adamA.GetState(), loaded.Optimizer.GetState());
            var original = File.ReadAllBytes(path);
            Check.Throws<ArgumentException>(() => TextCheckpoint.Save(path, b, tokenizer, progress));
            Check.Throws<ArgumentException>(() => TextCheckpoint.Save(path, b, tokenizer, progress, adamA));
            Check.True(original.SequenceEqual(File.ReadAllBytes(path)), "Rejected save damaged checkpoint.");
            void Tamper(Action<JsonNode> change)
            {
                var node = JsonNode.Parse(original)!; change(node["payload"]!);
                node["sha256"] = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(node["payload"]!.ToJsonString()))).ToLowerInvariant();
                File.WriteAllText(path, node.ToJsonString());
                Check.Throws<InvalidDataException>(() => TextCheckpoint.Load(path));
            }
            Tamper(p => p["optimizer"]!["secondMoment"]![0] = -1);
            Tamper(p => p["optimizer"]!["firstMoment"]!.AsArray().RemoveAt(0));
            Tamper(p => p["optimizer"]!["options"]!["beta2"] = 1);
            Tamper(p => p["optimizer"]!["step"] = -1);
            Tamper(p => p["optimizer"] = null);
            // Construct a true v1 envelope from an SGD save: old files remain readable.
            TextCheckpoint.Save(path, Model(), tokenizer, new TextTrainingState(2, 42, 0.02, new string('b', 64)));
            var legacy = JsonNode.Parse(File.ReadAllText(path))!; legacy["version"] = 1;
            legacy["payload"]!.AsObject().Remove("optimizer");
            legacy["payload"]!["training"]!.AsObject().Remove("batchSize"); legacy["payload"]!["training"]!.AsObject().Remove("optimizer");
            legacy["sha256"] = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(legacy["payload"]!.ToJsonString()))).ToLowerInvariant();
            File.WriteAllText(path, legacy.ToJsonString());
            var v1 = TextCheckpoint.Load(path);
            Check.True(v1.Optimizer is null && v1.Training!.Optimizer == "sgd" && v1.Training.BatchSize == 1, "Legacy SGD checkpoint changed meaning.");
        }
        finally { Directory.Delete(root, true); }
    }
}
