using System.Globalization;
using NeuralNetworks;

CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
if (args.Length > 1 || (args.Length == 1 && args[0] == "--help"))
{
    Console.WriteLine("Usage: dotnet run --project examples/TextGeneration -- [one-character-prompt]");
    return args.Length > 1 ? 2 : 0;
}
try
{
    (string Prompt, string Reply)[] corpus = [("a", "ola"), ("b", "oi")];
    var tokenizer = CharacterTokenizer.FromCorpus(corpus.SelectMany(x => new[] { x.Prompt, x.Reply }));
    var model = new TextTransformer(tokenizer.VocabularySize, sourceLength: 2, targetLength: 4,
        modelSize: 4, headCount: 2, feedForwardSize: 8, encoderLayerCount: 2, decoderLayerCount: 2, seed: 42);
    var source = corpus.Select(x => tokenizer.Encode(x.Prompt, addEos: true)).ToArray();
    var target = corpus.Select(x => tokenizer.Encode(x.Reply, addEos: true)).ToArray();
    var custom = args.Length == 1 ? tokenizer.Encode(args[0], addEos: true) : null;
    if (custom is not null && custom.Length > model.SourceLength)
        throw new ArgumentException("This tiny demo accepts at most one Unicode scalar plus EOS as its prompt.");
    Console.WriteLine($"Character vocabulary: {tokenizer.VocabularySize}; encoder layers: {model.EncoderLayerCount}; decoder layers: {model.DecoderLayerCount}; heads: {model.HeadCount}");
    Console.WriteLine("Local training pairs: a -> ola, b -> oi. No downloads or pretrained weights.");
    double MeanLoss() => source.Select((s, i) => model.Loss(s, target[i])).Average();
    var initial = MeanLoss();
    for (var epoch = 0; epoch < 800; epoch++)
        for (var i = 0; i < source.Length; i++) model.TrainSample(source[i], target[i], learningRate: 0.05);
    var final = MeanLoss();
    Console.WriteLine($"Mean token cross-entropy: {initial:F6} -> {final:F6}");
    if (final >= initial / 10) throw new InvalidOperationException("Training loss did not decrease sufficiently.");
    for (var i = 0; i < source.Length; i++)
    {
        var generated = model.Generate(source[i]);
        var reply = tokenizer.Decode(generated.Tokens);
        Console.WriteLine($"{corpus[i].Prompt} -> {reply}; ids=[{string.Join(',', generated.Tokens)}]; EOS={generated.EndedWithEos}");
        if (!generated.EndedWithEos || reply != corpus[i].Reply)
            throw new InvalidOperationException("Greedy text generation did not reproduce the training pair.");
    }
    if (custom is not null)
    {
        var generated = model.Generate(custom);
        Console.WriteLine($"Custom prompt '{args[0]}' -> '{tokenizer.Decode(generated.Tokens)}'; EOS={generated.EndedWithEos}");
    }
    Console.WriteLine("Synthetic training-pair memorization only; this is not a general-purpose text model.");
    return 0;
}
catch (ArgumentException error) { Console.Error.WriteLine(error.Message); return 2; }
catch (Exception error) { Console.Error.WriteLine($"FAIL: {error.Message}"); return 1; }
