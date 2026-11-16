namespace NeuralNetworks.Internal;

/// <summary>A scalar reverse-mode tape: each node stores at most two local partial derivatives.</summary>
internal sealed class Tape
{
    private readonly List<Node> nodes = [];
    private readonly record struct Node(double Value, int Left, int Right, double DLeft, double DRight);

    public Scalar Leaf(double value) => Add(value, -1, -1, 0, 0);
    public double Value(int id) => nodes[id].Value;

    public Scalar Add(double value, int left, int right, double dLeft, double dRight)
    {
        nodes.Add(new Node(Guard.Finite(value), left, right, Guard.Finite(dLeft), Guard.Finite(dRight)));
        return new Scalar(this, nodes.Count - 1);
    }

    public double[] Backward(Scalar loss)
    {
        var gradients = new double[nodes.Count];
        gradients[loss.Id] = 1;
        for (var i = nodes.Count - 1; i >= 0; i--)
        {
            if (gradients[i] == 0) continue;
            var node = nodes[i];
            Accumulate(node.Left, node.DLeft);
            Accumulate(node.Right, node.DRight);
            void Accumulate(int parent, double derivative)
            {
                if (parent >= 0)
                    gradients[parent] = Guard.Finite(gradients[parent] + Guard.Finite(gradients[i] * derivative));
            }
        }
        return gradients;
    }
}

internal readonly record struct Scalar(Tape Tape, int Id)
{
    public double Value => Tape.Value(Id);
    public Scalar Constant(double value) => Tape.Leaf(value);
    public static Scalar operator +(Scalar a, Scalar b) => Binary(a, b, a.Value + b.Value, 1, 1);
    public static Scalar operator -(Scalar a, Scalar b) => Binary(a, b, a.Value - b.Value, 1, -1);
    public static Scalar operator *(Scalar a, Scalar b) => Binary(a, b, a.Value * b.Value, b.Value, a.Value);
    public static Scalar operator +(Scalar a, double b) => a.Tape.Add(a.Value + b, a.Id, -1, 1, 0);
    public static Scalar operator -(double a, Scalar b) => b.Tape.Add(a - b.Value, b.Id, -1, -1, 0);
    public static Scalar operator *(Scalar a, double b) => a.Tape.Add(a.Value * b, a.Id, -1, b, 0);
    public static Scalar operator /(Scalar a, double b) => a * (1 / b);

    public Scalar Exp()
    {
        var value = Guard.Finite(Math.Exp(Value));
        return Tape.Add(value, Id, -1, value, 0);
    }

    public Scalar Sqrt()
    {
        var value = Math.Sqrt(Value);
        return Tape.Add(value, Id, -1, 0.5 / value, 0);
    }

    public Scalar Reciprocal() => Tape.Add(1 / Value, Id, -1, -1 / (Value * Value), 0);
    public Scalar Relu() => Tape.Add(Math.Max(0, Value), Id, -1, Value > 0 ? 1 : 0, 0);

    public Scalar Tanh()
    {
        var value = Math.Tanh(Value);
        return Tape.Add(value, Id, -1, 1 - value * value, 0);
    }

    public Scalar Sigmoid()
    {
        var exponential = Math.Exp(-Math.Abs(Value));
        var value = Value >= 0 ? 1 / (1 + exponential) : exponential / (1 + exponential);
        return Tape.Add(value, Id, -1, value * (1 - value), 0);
    }

    private static Scalar Binary(Scalar a, Scalar b, double value, double da, double db)
    {
        if (!ReferenceEquals(a.Tape, b.Tape)) throw new InvalidOperationException("Cannot mix evaluation tapes.");
        return a.Tape.Add(value, a.Id, b.Id, da, db);
    }
}
