import {
  ActivationFunction,
  EActivationFunction,
  IActivationResult,
} from './activation-functions';

interface INeuronOptions {
  activationFunction: EActivationFunction;
  /** Number of neurons in the previous layer — determines weight count. */
  previousLayerNeuronCount: number;
  /** Optional pre-set weights (used when restoring a saved network). */
  weights?: number[];
}

/** Fixed bias added to the weighted sum before activation. */
const BIAS = 0.25;

export default class Neuron {
  private readonly options: INeuronOptions;
  /** One weight per incoming connection. */
  public readonly weights: number[];

  constructor(options: INeuronOptions) {
    this.options = options;
    this.weights =
      options.weights ?? Neuron.randomWeights(options.previousLayerNeuronCount);
  }

  /** Xavier-style random initialisation in [-1, 1]. */
  private static randomWeights(count: number): number[] {
    return Array.from({ length: count }, () => Math.random() * 2 - 1);
  }

  /**
   * Computes the weighted sum of inputs plus bias.
   * Formula: Σ(xᵢ · wᵢ) + bias
   */
  private weightedSum(inputs: number[]): number {
    return (
      inputs.reduce((sum, x, i) => sum + x * this.weights[i], 0) + BIAS
    );
  }

  /**
   * Propagates inputs through this neuron.
   * @returns Activation result — the sigmoid value and whether the neuron fired.
   */
  public send(inputs: number[]): IActivationResult {
    if (inputs.length !== this.weights.length) {
      throw new Error(
        `Input length (${inputs.length}) does not match weight count (${this.weights.length}).`
      );
    }
    const z = this.weightedSum(inputs);
    return ActivationFunction.activate(this.options.activationFunction, z);
  }
}
