using NeuralNetworks;

internal static class TransformerTests
{
    public static void MultiHeadScores()
    {
        var model = new TransformerEncoder(2, 2, 4, 1, headCount: 2);
        var p = new double[model.ParameterCount];
        p[0] = 1; // input feature 0 -> embedding coordinate 0
        p[7] = 1; // input feature 1 -> embedding coordinate 2
        for (var i = 0; i < 4; i++) { p[12 + i * 5 + i] = 1; p[32 + i * 5 + i] = 1; }
        model.SetParameters(p);
        var heads = model.GetMultiHeadAttentionWeights([1, 0, 2, 3]);
        Check.True(heads.Length == 2 && model.HeadCount == 2, "Wrong head count.");
        var s00 = 2 / Math.Sqrt(2);
        var s01 = (2 + Math.Sin(1) + Math.Cos(1)) / Math.Sqrt(2);
        Check.Near(1 / (1 + Math.Exp(s01 - s00)), heads[0][0][0]);
        Check.Near(1 / (1 + Math.Exp((Math.Cos(0.01) - 1) / Math.Sqrt(2))), heads[1][0][0]);
        foreach (var head in heads)
            foreach (var row in head) Check.Near(1, row.Sum());
        Check.Same(heads[1][0], model.GetAttentionWeights([1, 0, 2, 3], headIndex: 1)[0]);
        for (var i = 0; i < 4; i++) { p[52 + i * 5 + i] = 1; p[72 + i * 5 + i] = 1; }
        p[132] = 1; // final head selects pooled coordinate zero
        for (var i = 0; i < 4; i++) { p[137 + i] = 1; p[145 + i] = 1; }
        model.SetParameters(p);
        // Independent arithmetic: concatenate the two weighted contexts, residual, two layer norms, pool, sigmoid.
        Check.Near(0.783837034923858, model.Predict([1, 0, 2, 3])[0]);
        Check.Throws<ArgumentOutOfRangeException>(() => model.GetAttentionWeights([1, 0, 2, 3], headIndex: 2));
        Check.Throws<ArgumentException>(() => new TransformerEncoder(2, 1, 4, 1, headCount: 3));
        Check.Throws<ArgumentOutOfRangeException>(() => new TransformerEncoder(2, 1, 4, 1, headCount: 0));
    }

    public static void Gradients()
    {
        var encoder = new TransformerEncoder(2, 1, 4, 2, 3, 7, headCount: 2);
        Numerical(encoder, () => encoder.Loss([0.2, 0.7], [0.3, 0.6]), () => encoder.ComputeGradient([0.2, 0.7], [0.3, 0.6]));
        var model = new TransformerSequenceToSequence(2, 2, 1, 1, 4, 2, 3, 7);
        Numerical(model, () => model.Loss([0.2, 0.7], [0.5, 0.3], [0.3, 0.6]),
            () => model.ComputeGradient([0.2, 0.7], [0.5, 0.3], [0.3, 0.6]));
        Numerical(model, () => model.Loss([0.2, 0.7], [0.5, 0.3], [0.3, 0.6], [false, true], [false, true]),
            () => model.ComputeGradient([0.2, 0.7], [0.5, 0.3], [0.3, 0.6], [false, true], [false, true]));
        var padded = new TransformerSequenceToSequence(3, 3, 1, 1, 4, 2, 3, 7);
        bool[] mask = [false, false, true];
        Numerical(padded, () => padded.Loss([0.2, 0.7, 0.9], [0.5, 0.3, 0.6], [0.3, 0.6, 0.2], mask, mask),
            () => padded.ComputeGradient([0.2, 0.7, 0.9], [0.5, 0.3, 0.6], [0.3, 0.6, 0.2], mask, mask));
    }

    private static void Numerical(IGradientNetwork model, Func<double> loss, Func<double[]> gradient)
    {
        var original = model.GetParameters();
        var analytic = gradient();
        Check.Same(original, model.GetParameters());
        for (var i = 0; i < original.Length; i++)
        {
            var changed = (double[])original.Clone();
            changed[i] += 1e-5;
            model.SetParameters(changed);
            var plus = loss();
            changed[i] -= 2e-5;
            model.SetParameters(changed);
            Check.Near((plus - loss()) / 2e-5, analytic[i], 3e-6);
        }
        model.SetParameters(original);
    }

    public static void CausalAndPaddingMasks()
    {
        var model = new TransformerSequenceToSequence(3, 3, 1, 1, 4, 2, 4, 42, encoderLayerCount: 2, decoderLayerCount: 2);
        var a = model.Predict([0.2, 0.7, 0.4], [0.5, 0.1, 0.2]);
        var b = model.Predict([0.2, 0.7, 0.4], [0.5, 0.1, 0.9]);
        Check.Near(a[0], b[0]);
        Check.Near(a[1], b[1]);
        bool[] padding = [false, false, true];
        Check.Throws<ArgumentException>(() => model.Predict([0.2, 0.7, 0.4], [0.5, 0.1, 0.2], sourcePaddingMask: [false, true, false]));
        Check.Throws<ArgumentException>(() => model.Predict([0.2, 0.7, 0.4], [0.5, 0.1, 0.2], targetPaddingMask: [false, true, false]));
        var maskedA = model.Predict([0.2, 0.7, 999], [0.5, 0.1, 999], padding, padding);
        var maskedB = model.Predict([0.2, 0.7, -999], [0.5, 0.1, -999], padding, padding);
        Check.Same(maskedA, maskedB);
        Check.Near(0, maskedA[2]);
        Check.Same(model.Generate([0.2, 0.7, 999], [0.5], padding), model.Generate([0.2, 0.7, -999], [0.5], padding));
        Check.Same(maskedA, model.Predict([0.2, 0.7, double.MaxValue], [0.5, 0.1, double.MaxValue], padding, padding));
        var maskedGradient = model.ComputeGradient([0.2, 0.7, 0], [0.5, 0.1, 0], [0.2, 0.4, 0], padding, padding);
        Check.Same(maskedGradient, model.ComputeGradient([0.2, 0.7, 0], [0.5, 0.1, 0], [0.2, 0.4, 1], padding, padding));
        Check.Near(model.Loss([0.2, 0.7, 0], [0.5, 0.1, 0], [0.2, 0.4, 0], padding, padding),
            model.Loss([0.2, 0.7, 0], [0.5, 0.1, 0], [0.2, 0.4, 1], padding, padding));
        var attention = model.GetAttentionWeights([0.2, 0.7, 999], [0.5, 0.1, 999], padding, padding);
        foreach (var head in attention.DecoderSelf)
            for (var query = 0; query < 3; query++)
            {
                Check.Near(1, head[query].Sum());
                for (var key = 0; key < 3; key++)
                    if (key > query || padding[key]) Check.Near(0, head[query][key]);
            }
        foreach (var head in attention.Cross)
            foreach (var row in head) { Check.Near(1, row.Sum()); Check.Near(0, row[2]); }
        foreach (var head in attention.EncoderSelf)
            foreach (var row in head) { Check.Near(1, row.Sum()); Check.Near(0, row[2]); }
        var layers = model.GetLayerAttentionWeights([0.2, 0.7, 999], [0.5, 0.1, 999], padding, padding);
        foreach (var layer in layers.DecoderSelf)
            foreach (var head in layer)
                for (var query = 0; query < 3; query++)
                    for (var key = 0; key < 3; key++)
                        if (key > query || padding[key]) Check.Near(0, head[query][key]);
        foreach (var layer in layers.Cross)
            foreach (var head in layer)
                foreach (var row in head) Check.Near(0, row[2]);
        attention.Cross[0][0][0] = 999;
        Check.True(model.GetAttentionWeights([0.2, 0.7, 0.4], [0.5, 0.1, 0.2]).Cross[0][0][0] != 999, "Attention was aliased.");
        var other = model.Predict([0.9, 0.1, 0.4], [0.5, 0.1, 0.2]);
        Check.True(Math.Abs(other[0] - a[0]) > 1e-6, "Decoder ignored encoder memory.");
    }

    public static void Boundaries()
    {
        var model = new TransformerSequenceToSequence(2, 2, 1, 1, 4, 2, 4, 42);
        var before = model.GetParameters();
        Check.Throws<ArgumentException>(() => model.Predict([0], [0.5, 0]));
        Check.Throws<ArgumentNullException>(() => model.Predict(null!, [0.5, 0]));
        Check.Throws<ArgumentException>(() => model.Predict([0, 1], [0.5]));
        Check.Throws<ArgumentException>(() => model.Predict([0, 1], [0.5, 0], [true, true]));
        Check.Throws<ArgumentException>(() => model.Predict([0, 1], [0.5, 0], targetPaddingMask: [true, false]));
        Check.Throws<ArgumentException>(() => model.Predict([0, 1], [0.5, 0], sourcePaddingMask: [false]));
        Check.Throws<ArgumentException>(() => model.TrainSample([0, 1], [0.5, 0], [0, 2], 0.1));
        Check.Throws<ArgumentOutOfRangeException>(() => model.TrainSample([0, 1], [0.5, 0], [0, 1], 0));
        Check.Throws<ArgumentException>(() => model.Generate([0, 1], []));
        Check.Same(before, model.GetParameters());
        var packed = new double[] { 0, 1, 0.5, 0 };
        Check.Same(((IGradientNetwork)model).Predict(packed), model.Predict([0, 1], [0.5, 0]));
        var duplicate = new TransformerSequenceToSequence(2, 2, 1, 1, 4, 2, 4, 42);
        model.TrainSample([0, 1], [0.5, 0], [1, 0], 0.1);
        duplicate.TrainSample([0, 1], [0.5, 0], [1, 0], 0.1);
        Check.Same(model.GetParameters(), duplicate.GetParameters());
        var huge = Enumerable.Repeat(double.MaxValue, model.ParameterCount).ToArray();
        model.SetParameters(huge);
        Check.Throws<ArithmeticException>(() => model.TrainSample([0.2, 0.7], [0.5, 0.2], [1, 0], 0.1));
        Check.Same(huge, model.GetParameters());
        Check.Throws<ArgumentException>(() => new TransformerSequenceToSequence(2, 2, 1, 1, 4, 3));
        Check.Throws<ArgumentOutOfRangeException>(() => new TransformerSequenceToSequence(0, 2, 1, 1));
        // Different source/target lengths and multiple numerical features are supported.
        var shaped = new TransformerSequenceToSequence(3, 2, 2, 2, 4, 2, 3, 42);
        var shapedOutput = shaped.Predict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.5, 0.5, 0.1, 0.2]);
        Check.True(shapedOutput.Length == 4 && shapedOutput.All(double.IsFinite), "Wrong sequence output shape.");
        Check.True(shaped.Generate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.5, 0.5]).Length == 4, "Wrong generation shape.");
    }

    public static void LearnsReverse()
    {
        var model = new TransformerSequenceToSequence(2, 2, 1, 1, 4, 2, 8, 42);
        double[][] source = [[0, 0], [0, 1], [1, 0], [1, 1]];
        double[][] target = [[0, 0], [1, 0], [0, 1], [1, 1]];
        double[][] shifted = target.Select(t => new[] { 0.5, t[0] }).ToArray();
        var initial = source.Select((s, i) => model.Loss(s, shifted[i], target[i])).Average();
        for (var epoch = 0; epoch < 1500; epoch++)
            for (var i = 0; i < source.Length; i++) model.TrainSample(source[i], shifted[i], target[i], 0.1);
        var final = source.Select((s, i) => model.Loss(s, shifted[i], target[i])).Average();
        Console.WriteLine($"  Encoder-decoder reversal: {initial:F6} -> {final:F6}");
        Check.True(final < initial / 10, "Sequence loss did not fall sufficiently.");
        for (var i = 0; i < source.Length; i++)
        {
            var generated = model.Generate(source[i], [0.5]);
            for (var j = 0; j < 2; j++) Check.Near(target[i][j], generated[j], 0.2);
        }
    }
}
