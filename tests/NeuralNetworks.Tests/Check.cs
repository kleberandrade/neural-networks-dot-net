internal static class Check
{
    public static void Near(double expected, double actual, double tolerance = 1e-9)
    {
        if (!double.IsFinite(actual) || Math.Abs(expected - actual) > tolerance)
            throw new Exception($"Expected {expected}, got {actual}; tolerance {tolerance}.");
    }

    public static void True(bool condition, string message)
    {
        if (!condition) throw new Exception(message);
    }

    public static void Same(double[] expected, double[] actual) =>
        True(expected.SequenceEqual(actual), "Arrays differ.");

    public static void Throws<T>(Action action) where T : Exception
    {
        try { action(); }
        catch (T) { return; }
        throw new Exception($"Expected {typeof(T).Name}.");
    }
}
