import { EActivationFunction } from './activation-functions';
import Neuron from './neuron';

interface INeuronalNetOptions {
  activationFunction: EActivationFunction;
  inputCount: number;
  outputCount: number;
  /** Array where each element is the neuron count for that hidden layer. */
  hiddenLayers: number[];
}

export interface INetworkOutput {
  /** Final output values from the output layer (sigmoid-activated, 0–1). */
  finalOutputs: number[];
  /**
   * Per-layer fired flags — indexed from the first hidden layer onward.
   * 1 = fired, 0 = did not fire.
   */
  firedNeurons: number[][];
  /**
   * Raw activation values at every layer including the input layer.
   * Index 0 = user inputs, index n = output layer values.
   */
  eachLayerInputValues: number[][];
}

export default class NeuronalNet {
  private readonly layers: Neuron[][];

  constructor(options: INeuronalNetOptions) {
    this.layers = NeuronalNet.buildLayers(options);
  }

  /**
   * Builds hidden + output layers.
   * Input layer neurons are implicit — they are not Neuron objects because
   * they simply pass values through without transformation.
   */
  private static buildLayers(options: INeuronalNetOptions): Neuron[][] {
    const layerSizes = [...options.hiddenLayers, options.outputCount];

    return layerSizes.map((neuronCount, layerIndex) => {
      const prevCount =
        layerIndex === 0
          ? options.inputCount
          : layerSizes[layerIndex - 1];

      return Array.from(
        { length: neuronCount },
        () =>
          new Neuron({
            activationFunction: options.activationFunction,
            previousLayerNeuronCount: prevCount,
          })
      );
    });
  }

  /**
   * Runs a forward pass through the network.
   * @param userInputs - Raw numeric inputs from the user (one per input neuron).
   */
  public send(userInputs: number[]): INetworkOutput {
    const eachLayerInputValues: number[][] = [userInputs];
    const firedNeurons: number[][] = [];

    let currentInputs = userInputs;

    for (const layer of this.layers) {
      const layerOutputs: number[] = [];
      const layerFired: number[] = [];

      for (const neuron of layer) {
        const result = neuron.send(currentInputs);
        layerOutputs.push(result.value);
        layerFired.push(result.fired);
      }

      firedNeurons.push(layerFired);
      currentInputs = layerOutputs;
      eachLayerInputValues.push(layerOutputs);
    }

    return {
      finalOutputs: currentInputs,
      firedNeurons,
      eachLayerInputValues,
    };
  }
}
