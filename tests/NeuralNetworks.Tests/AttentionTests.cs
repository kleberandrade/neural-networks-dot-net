using NeuralNetworks;

internal static class AttentionTests
{
    public static void KnownScaledAttention()
    {
        var model = new TransformerEncoder(2, 1, 2, 1, 2);
        var parameters = new double[model.ParameterCount];
        // Input projection [x] -> [x, 0]; Q and K are identities.
        parameters[0] = 1;
        parameters[4] = parameters[8] = 1;
        parameters[10] = parameters[14] = 1;
        model.SetParameters(parameters);
        var attention = model.GetAttentionWeights([1, 2]);
        // Embedded vectors: [1,1] and [2+sin(1),cos(1)].
        var second0 = 2 + Math.Sin(1);
        var second1 = Math.Cos(1);
        var s00 = 2 / Math.Sqrt(2);
        var s01 = (second0 + second1) / Math.Sqrt(2);
        var s11 = (second0 * second0 + second1 * second1) / Math.Sqrt(2);
        Check.Near(1 / (1 + Math.Exp(s01 - s00)), attention[0][0]);
        Check.Near(1 / (1 + Math.Exp(s11 - s01)), attention[1][0]);
        foreach (var row in attention)
        {
            Check.Near(1, row.Sum());
            Check.True(row.All(x => x >= 0 && x <= 1), "Invalid attention weights.");
        }
        var saved = attention[0][0];
        attention[0][0] = 999;
        Check.Near(saved, model.GetAttentionWeights([1, 2])[0][0]);
        var changed = model.GetAttentionWeights([1, 3]);
        Check.True(Math.Abs(changed[0][0] - saved) > 0.01, "Query did not attend to the other position.");
        Check.Throws<ArgumentException>(() => model.GetAttentionWeights([1]));
    }
}
