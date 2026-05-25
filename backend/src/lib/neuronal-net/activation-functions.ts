export enum EActivationFunction {
  SIGMOID
}

export interface IActivationResult {
  /** The raw sigmoid output value (0–1) */
  value: number;
  /** 1 if the neuron fired (sigmoid > 0.6), 0 otherwise */
  fired: 0 | 1;
}

export class ActivationFunction {
  /** Sigmoid squashes any real number into the range (0, 1). */
  public static sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  /**
   * Applies the chosen activation function and returns both the activation
   * value AND a "fired" flag for visualisation purposes.
   *
   * The raw sigmoid value is always forwarded to the next layer — un-fired
   * neurons still contribute their real output to downstream computation.
   */
  public static activate(
    activationFunction: EActivationFunction,
    x: number
  ): IActivationResult {
    switch (activationFunction) {
      case EActivationFunction.SIGMOID: {
        const value = ActivationFunction.sigmoid(x);
        const fired: 0 | 1 = value > 0.6 ? 1 : 0;
        return { value, fired };
      }
    }
  }
}
