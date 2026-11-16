using System.Globalization;

internal sealed class Options
{
    private readonly Dictionary<string, string> values = new(StringComparer.Ordinal);
    public static Options Parse(string[] args)
    {
        var result = new Options();
        for (var i = 0; i < args.Length; i++)
        {
            if (!args[i].StartsWith("--", StringComparison.Ordinal)) throw new ArgumentException("Expected --option, got " + args[i]);
            var key = args[i][2..];
            var value = key is "sample" or "new-corpus" or "overwrite" ? "true"
                : ++i < args.Length && !args[i].StartsWith("--", StringComparison.Ordinal) ? args[i] : throw new ArgumentException("Missing value for --" + key);
            if (!result.values.TryAdd(key, value)) throw new ArgumentException("Duplicate option --" + key);
        }
        return result;
    }
    public void Allow(params string[] keys)
    {
        foreach (var key in values.Keys)
            if (!keys.Contains(key, StringComparer.Ordinal)) throw new ArgumentException("Unknown option --" + key);
    }
    public bool Has(string key) => values.ContainsKey(key);
    public string Get(string key, string fallback) => values.GetValueOrDefault(key, fallback);
    public string Required(string key) => values.TryGetValue(key, out var value) ? value : throw new ArgumentException("Missing --" + key);
    public int Integer(string key, int fallback) => values.TryGetValue(key, out var value) ? int.Parse(value, CultureInfo.InvariantCulture) : fallback;
    public double Number(string key, double fallback) => values.TryGetValue(key, out var value) ? double.Parse(value, CultureInfo.InvariantCulture) : fallback;
}
