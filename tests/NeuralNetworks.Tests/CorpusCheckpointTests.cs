using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Security.Cryptography;
using NeuralNetworks;

internal static class CorpusCheckpointTests
{
    private static string Scratch() { var path = Path.Combine(Path.GetTempPath(), "neural-tests-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(path); return path; }

    public static void CorpusFiles()
    {
        var root = Scratch();
        try
        {
            var path = Path.Combine(root, "pairs.jsonl");
            File.WriteAllText(path, "{\"prompt\":\"ola\",\"response\":\"oi 🌍\"}\n{\"prompt\":\"lua\",\"response\":\"noite\"}\n", new UTF8Encoding(false));
            var corpus = TextCorpus.LoadPairs(path);
            Check.True(corpus.Pairs.Count == 2, "Pair count differs.");
            var tokenizer = corpus.CreateTokenizer(); var encoded = corpus.Encode(tokenizer);
            Check.True(tokenizer.Decode(encoded[0].Target) == "oi 🌍", "Corpus encoding lost Unicode.");
            Check.True(encoded[0].Target[^1] == CharacterTokenizer.EosId, "No target EOS.");
            var again = TextCorpus.LoadPairs(path);
            Check.True(corpus.Fingerprint == again.Fingerprint, "Corpus hash not stable.");
            File.WriteAllText(path, " {\"response\":\"oi 🌍\", \"prompt\":\"ola\"}\n\n{\"response\":\"noite\",\"prompt\":\"lua\"}\n");
            Check.True(corpus.Fingerprint == TextCorpus.LoadPairs(path).Fingerprint, "JSON formatting changed corpus identity.");
            File.WriteAllText(path, "{\"prompt\":\"lua\",\"response\":\"noite\"}\n{\"prompt\":\"ola\",\"response\":\"oi 🌍\"}\n");
            Check.True(corpus.Fingerprint != TextCorpus.LoadPairs(path).Fingerprint, "Sample order did not change corpus identity.");
            Check.Throws<ArgumentException>(() => corpus.Encode(CharacterTokenizer.FromCorpus(["x"])));
            Check.Throws<ArgumentException>(() => corpus.Encode(tokenizer, sourceLength: 2));
            File.WriteAllText(path, "{\"prompt\":\"a\",\"response\":\"b\",\"typo\":true}\n");
            Check.Throws<InvalidDataException>(() => TextCorpus.LoadPairs(path));
            File.WriteAllText(path, "{\"prompt\":\"a\"}\n");
            Check.Throws<InvalidDataException>(() => TextCorpus.LoadPairs(path));
            File.WriteAllText(path, "{\"prompt\":\"a\",\"prompt\":\"b\",\"response\":\"c\"}\n");
            Check.Throws<InvalidDataException>(() => TextCorpus.LoadPairs(path));
            File.WriteAllBytes(path, [0xff, 0xfe, 0xff]);
            Check.Throws<InvalidDataException>(() => TextCorpus.LoadPairs(path));
            File.WriteAllText(path, "abcdef 🌍\n");
            var windows = TextCorpus.LoadText(path, promptLength: 2, responseLength: 2, stride: 2);
            Check.True(windows.Pairs[0] == new TextPair("ab", "cd") && windows.Pairs[1] == new TextPair("cd", "ef"), "Wrong text windows.");
            Check.Throws<InvalidDataException>(() => TextCorpus.LoadText(path, 100, 100));
            Check.Throws<InvalidDataException>(() => TextCorpus.LoadText(path, 2, 2, limits: new TextCorpusLimits(MaxBytes: 2)));
            Check.Throws<ArgumentOutOfRangeException>(() => TextCorpus.LoadText(path, 0, 2));
            Check.Throws<InvalidDataException>(() => TextCorpus.LoadText(path, 1, 1, limits: new TextCorpusLimits(MaxSamples: 1)));
        }
        finally { Directory.Delete(root, true); }
    }

    public static void CheckpointRoundtrip()
    {
        var root = Scratch();
        try
        {
            var tokenizer = CharacterTokenizer.FromCorpus(["ola 🌍", "oi"]);
            var model = new TextTransformer(tokenizer.VocabularySize, 6, 6, 4, 2, 3, 2, 2, 42);
            var source = tokenizer.Encode("ola", addEos: true); var target = tokenizer.Encode("oi", addEos: true);
            model.TrainSample(source, target, 0.01);
            var before = model.Predict(source, [CharacterTokenizer.BosId]);
            var path = Path.Combine(root, "model.json");
            var state = new TextTrainingState(3, 7, 0.01, new string('a', 64));
            TextCheckpoint.Save(path, model, tokenizer, state);
            var loaded = TextCheckpoint.Load(path);
            Check.Same(model.GetParameters(), loaded.Model.GetParameters());
            Check.True(tokenizer.GetVocabulary().SequenceEqual(loaded.Tokenizer.GetVocabulary()), "Tokenizer mapping differs.");
            Check.True(loaded.Training == state && loaded.Model.FeedForwardSize == 3, "Configuration/progress missing.");
            Check.Same(before[0], loaded.Model.Predict(source, [CharacterTokenizer.BosId])[0]);
            Check.Near(model.Loss(source, target), loaded.Model.Loss(source, target));
            Check.True(model.Generate(source).Tokens.SequenceEqual(loaded.Model.Generate(source).Tokens), "Generation changed after load.");
            model.TrainSample(source, target, 0.01); loaded.Model.TrainSample(source, target, 0.01);
            Check.Same(model.GetParameters(), loaded.Model.GetParameters());
            var bytes = File.ReadAllBytes(path);
            Check.Throws<ArgumentException>(() => TextCheckpoint.Save(path, model, CharacterTokenizer.FromCorpus(["z"])));
            Check.True(bytes.SequenceEqual(File.ReadAllBytes(path)), "Rejected save replaced checkpoint.");
            var p = model.GetParameters(); p[0] = double.Epsilon; p[1] = -0.0; p[2] = double.MaxValue;
            model.SetParameters(p); TextCheckpoint.Save(path, model, tokenizer);
            var restored = TextCheckpoint.Load(path).Model.GetParameters();
            Check.True(p.Select(BitConverter.DoubleToInt64Bits).SequenceEqual(restored.Select(BitConverter.DoubleToInt64Bits)), "Double bits did not roundtrip.");
        }
        finally { Directory.Delete(root, true); }
    }

    public static void InvalidCheckpoints()
    {
        var root = Scratch();
        try
        {
            var path = Path.Combine(root, "model.json");
            var tokenizer = CharacterTokenizer.FromCorpus(["abc"]);
            var model = new TextTransformer(tokenizer.VocabularySize, 3, 3, 4, 2, 3, 2, 2, 42);
            TextCheckpoint.Save(path, model, tokenizer);
            var original = File.ReadAllText(path);
            var broken = JsonNode.Parse(original)!; broken["version"] = 99; File.WriteAllText(path, broken.ToJsonString());
            Check.Throws<InvalidDataException>(() => TextCheckpoint.Load(path));
            broken = JsonNode.Parse(original)!; broken["payload"]!["parameters"]![0] = 123.0; File.WriteAllText(path, broken.ToJsonString());
            Check.Throws<InvalidDataException>(() => TextCheckpoint.Load(path));
            void Change(Action<JsonNode> change)
            {
                var node = JsonNode.Parse(original)!; change(node["payload"]!);
                var payload = node["payload"]!.ToJsonString(); node["sha256"] = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(payload))).ToLowerInvariant();
                File.WriteAllText(path, node.ToJsonString());
            }
            Change(p => p["configuration"]!["modelSize"] = int.MaxValue);
            Check.Throws<InvalidDataException>(() => TextCheckpoint.Load(path));
            Change(p => p["vocabulary"]![4] = "<bad>");
            Check.Throws<InvalidDataException>(() => TextCheckpoint.Load(path));
            Change(p => p["parameters"]!.AsArray().RemoveAt(0));
            Check.Throws<InvalidDataException>(() => TextCheckpoint.Load(path));
            Change(p => p["configuration"]!["unexpected"] = true);
            Check.Throws<InvalidDataException>(() => TextCheckpoint.Load(path));
            Change(p => p["configuration"] = null);
            Check.Throws<InvalidDataException>(() => TextCheckpoint.Load(path));
            Change(p => p["training"] = new JsonObject
            {
                ["epochsCompleted"] = -1,
                ["shuffleSeed"] = 42,
                ["learningRate"] = 0.01,
                ["corpusFingerprint"] = new string('a', 64)
            });
            Check.Throws<InvalidDataException>(() => TextCheckpoint.Load(path));
            File.WriteAllText(path, original.Replace("\"version\":3", "\"version\":3,\"version\":3"));
            Check.Throws<InvalidDataException>(() => TextCheckpoint.Load(path));
            File.WriteAllText(path, original);
            Check.Throws<InvalidDataException>(() => TextCheckpoint.Load(path, new TextCheckpointLimits(MaxParameters: 1)));
            Check.Throws<InvalidDataException>(() => TextCheckpoint.Load(path, new TextCheckpointLimits(MaxFileBytes: 2)));
            File.WriteAllText(path, "{\"format\":");
            Check.Throws<InvalidDataException>(() => TextCheckpoint.Load(path));
            Check.Throws<ArgumentException>(() => CharacterTokenizer.FromVocabulary(["x", "<bos>", "<eos>", "<unk>", "a"]));
        }
        finally { Directory.Delete(root, true); }
    }

    public static void EpochResume()
    {
        var root = Scratch();
        try
        {
            var path = Path.Combine(root, "pairs.jsonl");
            File.WriteAllText(path, "{\"prompt\":\"a\",\"response\":\"b\"}\n{\"prompt\":\"b\",\"response\":\"a\"}\n");
            var corpus = TextCorpus.LoadPairs(path); var tokenizer = corpus.CreateTokenizer(); var samples = corpus.Encode(tokenizer);
            var a = new TextTransformer(tokenizer.VocabularySize, 2, 2, 4, 2, 3, 1, 1, 7);
            var b = new TextTransformer(tokenizer.VocabularySize, 2, 2, 4, 2, 3, 1, 1, 7);
            TextTraining.TrainEpoch(a, samples, 0.01, 42, 0); TextTraining.TrainEpoch(a, samples, 0.01, 42, 1);
            TextTraining.TrainEpoch(b, samples, 0.01, 42, 0);
            var checkpoint = Path.Combine(root, "resume.json");
            TextCheckpoint.Save(checkpoint, b, tokenizer, new TextTrainingState(1, 42, 0.01, corpus.Fingerprint));
            var loaded = TextCheckpoint.Load(checkpoint);
            TextTraining.TrainEpoch(loaded.Model, samples, loaded.Training!.LearningRate, loaded.Training.ShuffleSeed, loaded.Training.EpochsCompleted);
            Check.Same(a.GetParameters(), loaded.Model.GetParameters());
            var before = b.GetParameters();
            Check.Throws<ArgumentException>(() => TextTraining.TrainEpoch(b, [samples[0], new EncodedTextPair([4], [4])], 0.01, 42, 1));
            Check.Same(before, b.GetParameters());
        }
        finally { Directory.Delete(root, true); }
    }

    public static void Sampling()
    {
        var model = new TextTransformer(6, 2, 3, 4, 2, 3, 2, 2, 42);
        var p = new double[model.ParameterCount]; var head = p.Length - 6 * 5;
        p[head + 4 * 5 + 4] = 10; p[head + 5 * 5 + 4] = 10; model.SetParameters(p);
        var options = new TextSamplingOptions(Temperature: 0.8, TopK: 2, Seed: 42, MaxNewTokens: 3);
        var a = model.Sample([4, 2], options); var b = model.Sample([4, 2], options);
        Check.True(a.Tokens.SequenceEqual(b.Tokens) && a.Tokens.All(id => id is 4 or 5), "Sample seed or top-k ignored.");
        Check.True(model.Sample([4, 2], options with { TopK = 1 }).Tokens.All(id => id == 4), "Top-1 tie behavior differs.");
        var diversity = Enumerable.Range(0, 20).Select(seed => string.Join(',', model.Sample([4, 2], options with { Seed = seed }).Tokens)).Distinct().Count();
        Check.True(diversity > 1, "Different seeds never changed sampling.");
        Check.Throws<ArgumentOutOfRangeException>(() => model.Sample([4], options with { Temperature = 0 }));
        Check.Throws<ArgumentOutOfRangeException>(() => model.Sample([4], options with { TopK = 99 }));
        Check.Throws<ArgumentOutOfRangeException>(() => model.Sample([4], options with { MaxNewTokens = 4 }));
        p[head + 1 * 5 + 4] = 3000; model.SetParameters(p);
        Check.True(model.Sample([4], options).Tokens.All(id => id is 4 or 5), "Excluded-token mass corrupted sampling.");
    }
}
