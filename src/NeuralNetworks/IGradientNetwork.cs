namespace NeuralNetworks;

/// <summary>Finite-vector sigmoid models trained by squared-error gradient descent.</summary>
public interface IGradientNetwork
{
    int InputSize { get; }
    int OutputSize { get; }
    int ParameterCount { get; }
    double[] Predict(double[] input);
    double Loss(double[] input, double[] target);
    double[] ComputeGradient(double[] input, double[] target);
    double TrainSample(double[] input, double[] target, double learningRate);
    double[] GetParameters();
    void SetParameters(double[] values);
}
