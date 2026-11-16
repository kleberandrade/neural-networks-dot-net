using System.Globalization;
using NeuralNetworks;

internal static class CorpusCommands
{
    public static int Run(string[] args)
    {
        if (args.Contains("--help")) { Usage(); return 0; }
        try
        {
            var command = args.Length == 0 ? "demo" : args[0]; var options = Options.Parse(args.Skip(1).ToArray());
            switch (command)
            {
                case "demo": Demo(options); break;
                case "train": Train(options, resume: false); break;
                case "resume": Train(options, resume: true); break;
                case "generate": Generate(options); break;
                default: throw new ArgumentException("Unknown command: " + command);
            }
            return 0;
        }
        catch (Exception error) when (error is ArgumentException or IOException or FormatException or OverflowException)
        { Console.Error.WriteLine("Error: " + error.Message); return 2; }
        catch (Exception error) { Console.Error.WriteLine("FAIL: " + error.Message); return 1; }
    }

    private static void Demo(Options options)
    {
        options.Allow("checkpoint", "epochs");
        var corpus = TextCorpus.LoadPairs(Path.Combine(AppContext.BaseDirectory, "Data", "words.jsonl"));
        var tokenizer = corpus.CreateTokenizer(); var data = corpus.Encode(tokenizer);
        var model = NewModel(tokenizer, data, 16, 2, 16, 2, 2, 42);
        var path = options.Get("checkpoint", Path.Combine("artifacts", "corpus-demo.json"));
        if (options.Has("checkpoint") && File.Exists(path)) throw new ArgumentException("Demo refuses an existing custom checkpoint; choose a new path.");
        var initial = MeanLoss(model, data);
        Fit(model, tokenizer, corpus, data, new TextTrainingState(0, 42, 0.02, corpus.Fingerprint), options.Integer("epochs", 500), path);
        var restored = TextCheckpoint.Load(path);
        Require(model.GetParameters().SequenceEqual(restored.Model.GetParameters()), "Checkpoint changed parameter values.");
        var matched = 0; var distinct = new HashSet<string>(StringComparer.Ordinal); var groups = corpus.Pairs.GroupBy(p => p.Prompt).ToArray();
        foreach (var group in groups)
        {
            var source = tokenizer.Encode(group.Key, addEos: true); var before = model.Generate(source); var after = restored.Model.Generate(source);
            Require(before.Tokens.SequenceEqual(after.Tokens) && before.EndedWithEos == after.EndedWithEos, "Reload changed generation.");
            var text = restored.Tokenizer.Decode(after.Tokens); var correct = after.EndedWithEos && group.Any(p => p.Response == text);
            if (correct) { matched++; distinct.Add(text); }
            Console.WriteLine($"{group.Key} -> {text}; EOS={after.EndedWithEos}; matches training alternatives={correct}");
        }
        var final = MeanLoss(restored.Model, data);
        Console.WriteLine($"Samples={data.Length}; prompts={groups.Length}; matched={matched}; distinct matched replies={distinct.Count}; mean sample CE={initial:F6} -> {final:F6}");
        Require(final < initial / 5 && matched >= Math.Ceiling(groups.Length * 0.8) && distinct.Count >= 8,
            "Corpus demo did not meet its declared learning/diversity criteria.");
        foreach (var seed in new[] { 7, 19, 42, 101 })
        {
            var sampled = restored.Model.Sample(tokenizer.Encode("?", addEos: true), new TextSamplingOptions(1, 4, seed));
            Console.WriteLine($"sample '?' seed={seed} -> {tokenizer.Decode(sampled.Tokens)}; EOS={sampled.EndedWithEos}");
        }
        Console.WriteLine("Reload verified. Checkpoint: " + Path.GetFullPath(path));
        Console.WriteLine("Training-corpus memorization metrics only; sampled text can be wrong and does not establish generalization.");
    }

    private static void Train(Options options, bool resume)
    {
        options.Allow("corpus", "checkpoint", "epochs", "format", "prompt-length", "response-length", "stride", "max-samples", "max-bytes", "max-field",
            "rate", "seed", "new-corpus", "overwrite", "model-size", "heads", "ff", "encoder-layers", "decoder-layers");
        var path = options.Required("checkpoint"); var corpus = ReadCorpus(options);
        TextTransformer model; CharacterTokenizer tokenizer; TextTrainingState state; EncodedTextPair[] data;
        if (resume)
        {
            foreach (var key in new[] { "model-size", "heads", "ff", "encoder-layers", "decoder-layers", "overwrite" })
                if (options.Has(key)) throw new ArgumentException("Resume cannot change architecture: --" + key);
            var loaded = TextCheckpoint.Load(path); model = loaded.Model; tokenizer = loaded.Tokenizer;
            data = corpus.Encode(tokenizer, model.SourceLength, model.TargetLength);
            var previous = loaded.Training ?? new TextTrainingState(0, 42, 0.02, corpus.Fingerprint);
            if (!options.Has("new-corpus") && !previous.CorpusFingerprint.Equals(corpus.Fingerprint, StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException("Resume corpus differs. Use --new-corpus to retain weights and reset epoch ordering; vocabulary stays fixed.");
            state = new TextTrainingState(options.Has("new-corpus") ? 0 : previous.EpochsCompleted,
                options.Integer("seed", previous.ShuffleSeed), options.Number("rate", previous.LearningRate), corpus.Fingerprint);
        }
        else
        {
            if (File.Exists(path) && !options.Has("overwrite")) throw new ArgumentException("Checkpoint already exists; choose a new path or --overwrite.");
            if (options.Has("new-corpus")) throw new ArgumentException("--new-corpus applies only to resume.");
            tokenizer = corpus.CreateTokenizer(); data = corpus.Encode(tokenizer);
            var seed = options.Integer("seed", 42);
            model = NewModel(tokenizer, data, options.Integer("model-size", 16), options.Integer("heads", 2), options.Integer("ff", 16),
                options.Integer("encoder-layers", 2), options.Integer("decoder-layers", 2), seed);
            state = new TextTrainingState(0, seed, options.Number("rate", 0.02), corpus.Fingerprint);
        }
        Fit(model, tokenizer, corpus, data, state, options.Integer("epochs", 200), path);
    }

    private static TextCorpus ReadCorpus(Options options)
    {
        var path = options.Required("corpus");
        var limits = new TextCorpusLimits(options.Integer("max-bytes", 8 * 1024 * 1024), options.Integer("max-samples", 10000), options.Integer("max-field", 4095));
        var format = options.Get("format", "pairs");
        if (format == "pairs" && new[] { "prompt-length", "response-length", "stride" }.Any(options.Has))
            throw new ArgumentException("Window settings require --format text.");
        return format switch
        {
            "pairs" => TextCorpus.LoadPairs(path, limits),
            "text" => TextCorpus.LoadText(path, options.Integer("prompt-length", 4), options.Integer("response-length", 4), options.Integer("stride", 4), limits),
            _ => throw new ArgumentException("--format must be pairs or text.")
        };
    }

    private static void Fit(TextTransformer model, CharacterTokenizer tokenizer, TextCorpus corpus, EncodedTextPair[] data,
        TextTrainingState state, int epochs, string path)
    {
        if (epochs <= 0 || !double.IsFinite(state.LearningRate) || state.LearningRate <= 0) throw new ArgumentOutOfRangeException(nameof(epochs));
        var end = checked(state.EpochsCompleted + epochs);
        Console.WriteLine($"Samples={data.Length}; vocabulary={tokenizer.VocabularySize}; source capacity={model.SourceLength}; target capacity={model.TargetLength}; parameters={model.ParameterCount}");
        Console.WriteLine($"Initial mean sample token CE: {MeanLoss(model, data):F6}; completed epochs={state.EpochsCompleted}");
        TextCheckpoint.Save(path, model, tokenizer, state);
        for (var epoch = state.EpochsCompleted; epoch < end; epoch++)
        {
            TextTraining.TrainEpoch(model, data, state.LearningRate, state.ShuffleSeed, epoch);
            if ((epoch + 1) % 50 == 0 || epoch + 1 == end)
            {
                TextCheckpoint.Save(path, model, tokenizer, state with { EpochsCompleted = epoch + 1 });
                Console.WriteLine($"Epoch {epoch + 1}: mean sample token CE={MeanLoss(model, data):F6}; checkpoint saved");
            }
        }
    }

    private static void Generate(Options options)
    {
        options.Allow("checkpoint", "prompt", "sample", "temperature", "top-k", "seed", "max-tokens");
        if (!options.Has("sample") && (options.Has("temperature") || options.Has("top-k") || options.Has("seed")))
            throw new ArgumentException("Sampling settings require --sample.");
        var loaded = TextCheckpoint.Load(options.Required("checkpoint")); var prompt = options.Required("prompt");
        var source = loaded.Tokenizer.Encode(prompt, addEos: true); int? limit = options.Has("max-tokens") ? options.Integer("max-tokens", 1) : null;
        var result = options.Has("sample")
            ? loaded.Model.Sample(source, new TextSamplingOptions(options.Number("temperature", 1), options.Integer("top-k", 0), options.Integer("seed", 42), limit))
            : loaded.Model.Generate(source, limit);
        Console.WriteLine(loaded.Tokenizer.Decode(result.Tokens));
        Console.WriteLine($"EOS={result.EndedWithEos}; generated IDs=[{string.Join(',', result.Tokens)}]");
    }

    private static TextTransformer NewModel(CharacterTokenizer tokenizer, EncodedTextPair[] data, int size, int heads, int ff, int encoders, int decoders, int seed) =>
        new(tokenizer.VocabularySize, data.Max(p => p.Source.Length), data.Max(p => p.Target.Length), size, heads, ff, encoders, decoders, seed);
    private static double MeanLoss(TextTransformer model, EncodedTextPair[] data) => data.Select(p => model.Loss(p.Source, p.Target)).Average();
    private static void Require(bool value, string message) { if (!value) throw new InvalidOperationException(message); }

    private static void Usage() => Console.WriteLine("""
        Commands (UTF-8 corpus, .NET 10):
          demo [--checkpoint PATH] [--epochs 500]
          train --corpus PATH --checkpoint PATH [--epochs 200] [--format pairs|text]
          resume --corpus PATH --checkpoint PATH [--epochs 200] [--new-corpus]
          generate --checkpoint PATH --prompt TEXT [--sample --temperature 1 --top-k 4 --seed 42]
        Training settings: --rate --seed --model-size --heads --ff --encoder-layers --decoder-layers
        Plain-text windows: --prompt-length --response-length --stride
        Corpus limits: --max-bytes --max-samples --max-field; generation limit: --max-tokens
        Train refuses an existing checkpoint unless --overwrite. Demo replaces its default artifact.
        """);
}
