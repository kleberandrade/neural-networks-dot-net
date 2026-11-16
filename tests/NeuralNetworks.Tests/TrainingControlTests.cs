using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using NeuralNetworks;

internal static class TrainingControlTests
{
    private static TextTransformer Model() => new(7, 3, 4, 4, 2, 3, 1, 1, 7);
    private static readonly EncodedTextPair[] Samples = [new([4, 2], [5, 2]), new([5, 4, 2], [6, 5, 2]), new([6, 2], [4, 2])];
    private static void SameState(AdamState a, AdamState b)
    {
        Check.True(a.Options == b.Options && a.Step == b.Step, "Training policy or step differs.");
        Check.Same(a.FirstMoment, b.FirstMoment); Check.Same(a.SecondMoment, b.SecondMoment);
    }

    public static void GlobalClipping()
    {
        double[] original = [3, 4]; var clipped = GradientClipping.ClipGlobalNorm(original, 2);
        Check.Near(1.2, clipped[0]); Check.Near(1.6, clipped[1]); Check.Same([3, 4], original);
        Check.Same(original, GradientClipping.ClipGlobalNorm(original, 5));
        Check.Same([0, 0], GradientClipping.ClipGlobalNorm([0, 0], 1));
        var huge = GradientClipping.ClipGlobalNorm([double.MaxValue, -double.MaxValue], 2);
        Check.Near(Math.Sqrt(2), huge[0]); Check.Near(-Math.Sqrt(2), huge[1]);
        Check.Near(1e-100, GradientClipping.ClipGlobalNorm([double.MaxValue], 1e-100)[0], 1e-112);
        Check.Throws<ArgumentOutOfRangeException>(() => GradientClipping.ClipGlobalNorm([1], 0));
        Check.Throws<ArgumentOutOfRangeException>(() => GradientClipping.ClipGlobalNorm([1], double.PositiveInfinity));
        Check.Throws<ArgumentException>(() => GradientClipping.ClipGlobalNorm([double.NaN], 1));
        Check.Throws<ArgumentException>(() => GradientClipping.ClipGlobalNorm([], 1));
        clipped[0] = 99; Check.Near(3, original[0]);
    }

    public static void ScheduleValues()
    {
        var schedule = new LinearWarmupDecay(2, 4, 0.2);
        double[] expected = [0.005, 0.01, 0.008, 0.006, 0.004, 0.002, 0.002];
        for (var i = 0; i < expected.Length; i++) Check.Near(expected[i], schedule.GetLearningRate(0.01, i + 1), 1e-14);
        var noWarmup = new LinearWarmupDecay(0, 2, 0.5);
        Check.Near(0.01, noWarmup.GetLearningRate(0.01, 1)); Check.Near(0.0075, noWarmup.GetLearningRate(0.01, 2));
        Check.Near(0.005, noWarmup.GetLearningRate(0.01, long.MaxValue));
        Check.Throws<ArgumentOutOfRangeException>(() => schedule.GetLearningRate(0.01, 0));
        Check.Throws<ArgumentOutOfRangeException>(() => new LinearWarmupDecay(-1, 4).GetLearningRate(0.01, 1));
        Check.Throws<ArgumentOutOfRangeException>(() => new LinearWarmupDecay(0, 0).GetLearningRate(0.01, 1));
        Check.Throws<ArgumentOutOfRangeException>(() => new LinearWarmupDecay(0, 4, 0).GetLearningRate(0.01, 1));
        Check.Throws<ArgumentOutOfRangeException>(() => new LinearWarmupDecay(0, 4, double.NaN).GetLearningRate(0.01, 1));
    }

    public static void BatchAndAtomicity()
    {
        var actual = Model(); var expected = Model();
        var policy = new AdamOptions(MaxGradientNorm: 0.05, Schedule: new LinearWarmupDecay(2, 4, 0.2));
        var optimizer = new AdamOptimizer(actual, policy);
        var gradients = Samples.Take(2).Select(s => expected.ComputeGradient(s.Source, s.Target)).ToArray();
        var mean = Enumerable.Range(0, expected.ParameterCount).Select(i => gradients.Sum(g => g[i] / 2)).ToArray();
        var norm = Math.Sqrt(mean.Sum(x => x * x)); Check.True(norm > 0.05, "Fixture does not exercise clipping.");
        var independentlyClipped = mean.Select(x => x * (0.05 / norm)).ToArray();
        var expectedAdam = new AdamOptimizer(expected);
        expectedAdam.Step(independentlyClipped, 0.005);
        Check.Near(0.005, optimizer.GetNextLearningRate(0.01));
        TextTraining.TrainBatch(actual, Samples.Take(2).ToArray(), optimizer, 0.01);
        foreach (var pair in expected.GetParameters().Zip(actual.GetParameters())) Check.Near(pair.First, pair.Second, 1e-12);
        foreach (var pair in expectedAdam.GetState().FirstMoment.Zip(optimizer.GetState().FirstMoment)) Check.Near(pair.First, pair.Second, 1e-12);
        foreach (var pair in expectedAdam.GetState().SecondMoment.Zip(optimizer.GetState().SecondMoment)) Check.Near(pair.First, pair.Second, 1e-12);
        Check.Near(0.01, optimizer.GetNextLearningRate(0.01));
        var state = optimizer.GetState(); var weights = actual.GetParameters();
        var bad = new double[actual.ParameterCount]; bad[^1] = double.PositiveInfinity;
        Check.Throws<ArgumentException>(() => optimizer.Step(bad, 0.01));
        Check.Same(weights, actual.GetParameters()); SameState(state, optimizer.GetState());
        var underflow = new AdamOptimizer(actual, policy with { Schedule = new LinearWarmupDecay(long.MaxValue, 1) });
        Check.Throws<ArithmeticException>(() => underflow.Step(new double[actual.ParameterCount], double.Epsilon));
        Check.True(underflow.GetState().Step == 0, "Rejected rate advanced scheduler."); Check.Same(weights, actual.GetParameters());
        Check.Throws<ArgumentOutOfRangeException>(() => new AdamOptimizer(actual, policy with { MaxGradientNorm = -1 }));
        // A finite enormous raw gradient can be clipped before Adam squares it.
        var huge = new double[actual.ParameterCount]; huge[0] = double.MaxValue;
        optimizer.Step(huge, 0.01); Check.True(optimizer.GetState().Step == 2, "Clipping did not precede moments.");
    }

    public static void CheckpointResume()
    {
        var root = Path.Combine(Path.GetTempPath(), "controls-tests-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(root);
        try
        {
            var a = Model(); var b = Model();
            var options = new AdamOptions(MaxGradientNorm: 0.1, Schedule: new LinearWarmupDecay(3, 4, 0.2));
            var adamA = new AdamOptimizer(a, options); var adamB = new AdamOptimizer(b, options);
            for (var epoch = 0; epoch < 5; epoch++) TextTraining.TrainEpoch(a, Samples, adamA, 2, 0.003, 42, epoch);
            TextTraining.TrainEpoch(b, Samples, adamB, 2, 0.003, 42, 0);
            var path = Path.Combine(root, "controls.json"); var tokenizer = CharacterTokenizer.FromCorpus(["abc"]);
            var progress = new TextTrainingState(1, 42, 0.003, new string('a', 64), 2, "adam");
            TextCheckpoint.Save(path, b, tokenizer, progress, adamB); var original = File.ReadAllText(path);
            var loaded = TextCheckpoint.Load(path); SameState(adamB.GetState(), loaded.Optimizer!.GetState());
            Check.Near(0.003, loaded.Optimizer.GetNextLearningRate(loaded.Training!.LearningRate));
            for (var epoch = loaded.Training.EpochsCompleted; epoch < 5; epoch++)
                TextTraining.TrainEpoch(loaded.Model, Samples, loaded.Optimizer, 2, 0.003, 42, epoch);
            Check.Same(a.GetParameters(), loaded.Model.GetParameters()); SameState(adamA.GetState(), loaded.Optimizer.GetState());
            Check.Near(0.0006, loaded.Optimizer.GetNextLearningRate(0.003));
            void Change(Action<JsonNode> change)
            {
                var node = JsonNode.Parse(original)!; change(node["payload"]!);
                node["sha256"] = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(node["payload"]!.ToJsonString()))).ToLowerInvariant();
                File.WriteAllText(path, node.ToJsonString()); Check.Throws<InvalidDataException>(() => TextCheckpoint.Load(path));
            }
            Change(p => p["optimizer"]!["options"]!["maxGradientNorm"] = 0);
            Change(p => p["optimizer"]!["options"]!.AsObject().Remove("maxGradientNorm"));
            Change(p => p["optimizer"]!["options"]!["schedule"]!["decaySteps"] = 0);
            Change(p => p["optimizer"]!["options"]!["schedule"]!["warmupSteps"] = -1);
            Change(p => p["optimizer"]!["options"]!["schedule"]!.AsObject().Remove("minimumFactor"));
            // A v2 Adam checkpoint must still mean no clipping and a constant rate.
            var legacyModel = Model(); var legacyAdam = new AdamOptimizer(legacyModel);
            TextTraining.TrainEpoch(legacyModel, Samples, legacyAdam, 2, 0.003, 42, 0);
            TextCheckpoint.Save(path, legacyModel, tokenizer, progress, legacyAdam);
            var legacy = JsonNode.Parse(File.ReadAllText(path))!; legacy["version"] = 2;
            legacy["payload"]!["optimizer"]!["options"]!.AsObject().Remove("maxGradientNorm");
            legacy["payload"]!["optimizer"]!["options"]!.AsObject().Remove("schedule");
            legacy["sha256"] = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(legacy["payload"]!.ToJsonString()))).ToLowerInvariant();
            File.WriteAllText(path, legacy.ToJsonString()); var v2 = TextCheckpoint.Load(path);
            Check.True(v2.Optimizer!.GetState().Options == new AdamOptions(), "Legacy optimizer semantics changed.");
            Check.Near(0.003, v2.Optimizer.GetNextLearningRate(0.003));
            SameState(legacyAdam.GetState(), v2.Optimizer.GetState());
            TextTraining.TrainBatch(legacyModel, Samples, legacyAdam, 0.003);
            TextTraining.TrainBatch(v2.Model, Samples, v2.Optimizer, 0.003);
            Check.Same(legacyModel.GetParameters(), v2.Model.GetParameters()); SameState(legacyAdam.GetState(), v2.Optimizer.GetState());
        }
        finally { Directory.Delete(root, true); }
    }
}
