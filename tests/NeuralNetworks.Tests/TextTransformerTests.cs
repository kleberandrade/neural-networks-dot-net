using NeuralNetworks;

internal static class TextTransformerTests
{
    public static void Tokenizer()
    {
        var tokenizer = CharacterTokenizer.FromCorpus(["Olá 🌍!", "a b"]);
        var same = CharacterTokenizer.FromCorpus(["a b", "Olá 🌍!"]);
        Check.True(tokenizer.GetVocabulary().SequenceEqual(same.GetVocabulary()), "Vocabulary depends on corpus order.");
        var ids = tokenizer.Encode("Olá 🌍!", addBos: true, addEos: true);
        Check.True(ids[0] == CharacterTokenizer.BosId && ids[^1] == CharacterTokenizer.EosId, "Missing special tokens.");
        Check.True(ids.Length == 8, "Tokenizer split a supplementary Unicode scalar.");
        Check.True(tokenizer.Decode(ids) == "Olá 🌍!", "Unicode roundtrip failed.");
        Check.True(tokenizer.Encode("?")[0] == CharacterTokenizer.UnkId, "Unknown character not mapped.");
        Check.True(tokenizer.Decode([CharacterTokenizer.UnkId]) == "�", "Unknown decoding differs.");
        Check.True(tokenizer.Decode([CharacterTokenizer.BosId, ids[1], CharacterTokenizer.EosId, ids[2]]) == "O", "EOS did not stop decoding.");
        var vocabulary = tokenizer.GetVocabulary(); vocabulary[4] = "changed";
        Check.True(tokenizer.GetVocabulary()[4] != "changed", "Vocabulary was aliased.");
        Check.Throws<ArgumentException>(() => CharacterTokenizer.FromCorpus([]));
        Check.Throws<ArgumentNullException>(() => tokenizer.Encode(null!));
        Check.Throws<ArgumentException>(() => tokenizer.Encode("\ud800"));
        Check.Throws<ArgumentOutOfRangeException>(() => tokenizer.Decode([999]));
    }

    public static void StacksAndGradients()
    {
        var encoder = new TransformerEncoder(2, 1, 4, 1, 2, 7, 2, layerCount: 2);
        Check.True(encoder.LayerCount == 2 && encoder.GetLayerAttentionWeights([0.2, 0.7]).Length == 2, "Encoder stack not exposed.");
        Numerical(encoder, () => encoder.Loss([0.2, 0.7], [0.3]), () => encoder.ComputeGradient([0.2, 0.7], [0.3]));
        var numeric = new TransformerSequenceToSequence(2, 2, 1, 1, 4, 2, 2, 7, encoderLayerCount: 2, decoderLayerCount: 2);
        Check.True(numeric.EncoderLayerCount == 2 && numeric.DecoderLayerCount == 2, "Wrong stack depth.");
        Numerical(numeric, () => numeric.Loss([0.2, 0.7], [0.5, 0.3], [0.3, 0.6]),
            () => numeric.ComputeGradient([0.2, 0.7], [0.5, 0.3], [0.3, 0.6]));
        var text = new TextTransformer(5, 2, 2, 4, 2, 2, 2, 2, 7);
        Numerical(text, () => text.Loss([4, 2], [4, 2]), () => text.ComputeGradient([4, 2], [4, 2]));
        var signal = text.ComputeGradient([4, 2], [4, 2]);
        // Published layout: two 5x4 embedding matrices, two encoder blocks (118 each), two decoder blocks (206 each).
        foreach (var (start, count) in new[] { (40, 118), (158, 118), (276, 206), (482, 206) })
            Check.True(signal.Skip(start).Take(count).Any(g => Math.Abs(g) > 1e-10), "Gradient did not reach every layer.");
        Check.True(text.ParameterCount > new TextTransformer(5, 2, 2, 4, 2, 2, 1, 1, 7).ParameterCount, "Layers share their parameters.");
    }

    private static void Numerical(ParameterizedModel model, Func<double> loss, Func<double[]> gradient)
    {
        var original = model.GetParameters(); var analytic = gradient();
        Check.Same(original, model.GetParameters());
        for (var i = 0; i < original.Length; i++)
        {
            var changed = (double[])original.Clone(); changed[i] += 1e-5; model.SetParameters(changed); var plus = loss();
            changed[i] -= 2e-5; model.SetParameters(changed);
            Check.Near((plus - loss()) / 2e-5, analytic[i], 5e-6);
        }
        model.SetParameters(original);
    }

    public static void CategoricalAndGeneration()
    {
        var model = new TextTransformer(5, 2, 3, 4, 2, 2, 2, 2, 42);
        var p = new double[model.ParameterCount]; model.SetParameters(p);
        Check.Near(Math.Log(5), model.Loss([4, 2], [4, 2]));
        foreach (var row in model.Predict([4, 2], [1, 4]))
        {
            Check.Near(1, row.Sum()); foreach (var probability in row) Check.Near(0.2, probability);
        }
        var gradient = model.ComputeGradient([4, 2], [4, 2]);
        var head = p.Length - 5 * (4 + 1);
        for (var token = 0; token < 5; token++)
            Check.Near(0.2 - (token is 2 or 4 ? 0.5 : 0), gradient[head + token * 5 + 4]);
        p[head + 4 * 5 + 4] = -1000; model.SetParameters(p);
        Check.Near(500 + Math.Log(4), model.Loss([4, 2], [4, 2]));
        p[head + 4 * 5 + 4] = 1000; model.SetParameters(p);
        var limited = model.Generate([4, 2], maxNewTokens: 2);
        Check.True(!limited.EndedWithEos && limited.Tokens.SequenceEqual(new[] { 4, 4 }), "Generation ignored cap or IDs.");
        p[head + CharacterTokenizer.BosId * 5 + 4] = 3000; model.SetParameters(p);
        Check.True(model.Generate([4, 2], 2).Tokens.SequenceEqual(new[] { 4, 4 }), "Greedy allowed-token ranking underflowed.");
        p[head + CharacterTokenizer.BosId * 5 + 4] = 0;
        p[head + 2 * 5 + 4] = 2000; model.SetParameters(p);
        var ended = model.Generate([4, 2]);
        Check.True(ended.EndedWithEos && ended.Tokens.SequenceEqual(new[] { 2 }), "EOS did not terminate generation.");
    }

    public static void CausalityAndBoundaries()
    {
        var model = new TextTransformer(7, 4, 4, 4, 2, 4, 2, 2, 42);
        var first = model.Predict([4, 2], [1, 4, 5]);
        var future = model.Predict([4, 2], [1, 4, 6]);
        Check.Same(first[0], future[0]); Check.Same(first[1], future[1]);
        var padded = model.Predict([4, 2, 0, 0], [1, 4, 5, 0]);
        for (var i = 0; i < first.Length; i++) Check.Same(first[i], padded[i]);
        Check.Near(model.Loss([4, 2], [4, 2]), model.Loss([4, 2, 0], [4, 2, 0, 0]));
        Check.Same(model.ComputeGradient([4, 2], [4, 2]), model.ComputeGradient([4, 2, 0], [4, 2, 0, 0]));
        var attention = model.GetAttentionWeights([4, 2], [1, 4, 5]);
        Check.True(attention.EncoderSelf.Length == 2 && attention.DecoderSelf.Length == 2, "Missing stack snapshots.");
        foreach (var layer in attention.DecoderSelf)
            foreach (var head in layer)
                for (var query = 0; query < head.Length; query++)
                {
                    Check.Near(1, head[query].Sum());
                    for (var key = query + 1; key < head[query].Length; key++) Check.Near(0, head[query][key]);
                }
        Check.True(Math.Abs(first[0][4] - model.Predict([5, 2], [1, 4, 5])[0][4]) > 1e-6, "Text decoder ignored source.");
        var explicitShift = model.Predict([4, 2], [CharacterTokenizer.BosId, 4]);
        Check.Near(-(Math.Log(explicitShift[0][4]) + Math.Log(explicitShift[1][CharacterTokenizer.EosId])) / 2,
            model.Loss([4, 2], [4, 2]));
        var before = model.GetParameters();
        first[0][0] = 999; attention.Cross[0][0][0][0] = 999;
        Check.True(model.Predict([4, 2], [1])[0][0] != 999, "Probabilities were aliased.");
        Check.True(model.GetAttentionWeights([4, 2], [1]).Cross[0][0][0][0] != 999, "Attention was aliased.");
        Check.Throws<ArgumentException>(() => model.Predict([4, 2, 0, 0, 0], [1]));
        Check.Throws<ArgumentException>(() => model.Predict([4], [1, 1]));
        Check.Throws<ArgumentException>(() => model.Predict([4, 0, 2], [1]));
        Check.Throws<ArgumentException>(() => model.Predict([0, 0], [1]));
        Check.Throws<ArgumentOutOfRangeException>(() => model.Predict([7], [1]));
        Check.Throws<ArgumentException>(() => model.Predict([4], [4]));
        Check.Throws<ArgumentException>(() => model.Loss([4], [4]));
        Check.Throws<ArgumentException>(() => model.Loss([4], [2, 4, 2]));
        Check.Throws<ArgumentException>(() => model.Loss([4], [1, 2]));
        Check.Throws<ArgumentOutOfRangeException>(() => model.TrainSample([4, 2], [4, 2], 0));
        Check.Throws<ArgumentOutOfRangeException>(() => model.Generate([4], 5));
        Check.Same(before, model.GetParameters());
        Check.Throws<ArgumentOutOfRangeException>(() => new TransformerEncoder(2, 1, 4, 1, layerCount: 0));
        Check.Throws<ArgumentOutOfRangeException>(() => new TransformerSequenceToSequence(2, 2, 1, 1, encoderLayerCount: 0));
        Check.Throws<ArgumentOutOfRangeException>(() => new TextTransformer(7, 2, 2, decoderLayerCount: 0));
        var duplicate = new TextTransformer(7, 4, 4, 4, 2, 4, 2, 2, 42);
        model.TrainSample([4, 2], [4, 2], 0.01); duplicate.TrainSample([4, 2], [4, 2], 0.01);
        Check.Same(model.GetParameters(), duplicate.GetParameters());
        model.SetParameters(Enumerable.Repeat(double.MaxValue, model.ParameterCount).ToArray());
        var huge = model.GetParameters();
        Check.Throws<ArithmeticException>(() => model.TrainSample([4, 2], [4, 2], 0.01));
        Check.Same(huge, model.GetParameters());
    }

    public static void LearnsText()
    {
        var tokenizer = CharacterTokenizer.FromCorpus(["a", "b", "ola", "oi"]);
        var model = new TextTransformer(tokenizer.VocabularySize, 2, 4, 4, 2, 8, 2, 2, 42);
        int[][] source = [tokenizer.Encode("a", addEos: true), tokenizer.Encode("b", addEos: true)];
        int[][] target = [tokenizer.Encode("ola", addEos: true), tokenizer.Encode("oi", addEos: true)];
        var initial = source.Select((s, i) => model.Loss(s, target[i])).Average();
        for (var epoch = 0; epoch < 800; epoch++)
            for (var i = 0; i < source.Length; i++) model.TrainSample(source[i], target[i], 0.05);
        var final = source.Select((s, i) => model.Loss(s, target[i])).Average();
        Console.WriteLine($"  Text mean token CE: {initial:F6} -> {final:F6}");
        Check.True(final < initial / 10, "Text loss did not fall.");
        string[] expected = ["ola", "oi"];
        for (var i = 0; i < source.Length; i++)
        {
            var generated = model.Generate(source[i]);
            var text = tokenizer.Decode(generated.Tokens);
            Check.True(generated.EndedWithEos && text == expected[i], "Generated text differs: " + text);
        }
    }
}
