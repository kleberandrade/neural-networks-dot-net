using NeuralNetworks.Internal;

namespace NeuralNetworks;

/// <summary>Single-channel valid 2D cross-correlation, ReLU, global average pooling and sigmoid head.</summary>
public sealed class ConvolutionalNetwork : GradientNetwork
{
    private readonly LinearBlock kernels;
    private readonly LinearBlock output;
    public int Width { get; }
    public int Height { get; }
    public int KernelSize { get; }
    public int Filters { get; }

    public ConvolutionalNetwork(int width, int height, int kernelSize, int filters, int outputSize, int seed = 0)
        : base(checked(Guard.Positive(width, nameof(width)) * Guard.Positive(height, nameof(height))), outputSize, seed)
    {
        Width = width;
        Height = height;
        KernelSize = Guard.Positive(kernelSize, nameof(kernelSize));
        Filters = Guard.Positive(filters, nameof(filters));
        if (kernelSize > width || kernelSize > height) throw new ArgumentException("Kernel must fit inside the image.", nameof(kernelSize));
        kernels = AddLinear(checked(kernelSize * kernelSize), filters);
        output = AddLinear(filters, outputSize);
    }

    private protected override Scalar[] Forward(Scalar[] input, Scalar[] parameters)
    {
        var pooled = Operations.Zeros(input[0], Filters);
        var locations = (Width - KernelSize + 1) * (Height - KernelSize + 1);
        for (var row = 0; row <= Height - KernelSize; row++)
            for (var col = 0; col <= Width - KernelSize; col++)
            {
                var patch = new Scalar[KernelSize * KernelSize];
                for (var y = 0; y < KernelSize; y++)
                    for (var x = 0; x < KernelSize; x++)
                        patch[y * KernelSize + x] = input[(row + y) * Width + col + x];
                var activation = kernels.Apply(parameters, patch);
                for (var filter = 0; filter < Filters; filter++)
                    pooled[filter] += activation[filter].Relu() / locations;
            }
        return Operations.Sigmoid(output.Apply(parameters, pooled));
    }
}
