import { ActivationFunctionType } from './activation-functions/activation-function';
import { Layer } from './layer';
import { Neuron } from './neuron';

interface ModelType {
    layers: number[];
    thresholds: number[][];
    weights: number[][][];
}

interface BackpropagationConfig {
    epochs: number;
    activationFunction?: ActivationFunctionType;
    learningRate?: number;
}

'use strict';
export class Backpropagation {
    layers: Layer;
    epochs: number;
    activationFunction: ActivationFunctionType;
    error: number;

    /**
     * @construtor
     */
    constructor(
        config: BackpropagationConfig = {
            epochs: 1000,
            activationFunction: ActivationFunctionType.SIGMOIDAL,
            learningRate: 0.3
        }
    ) {
        this.error = 0;
        this.activationFunction = config.activationFunction;
        this.layers = new Layer(this.activationFunction, config.learningRate);
        this.epochs = config.epochs;
    }

    learn(dataset: Array<{ input: any; output: number }>) {
        this.datasetInputToFloatArray(dataset);

        this.runEpoch(dataset);
        let counterEpochs = 1;

        while (counterEpochs <= this.epochs) {
            counterEpochs++;

            if (counterEpochs % 1000 === 0) {
                console.log(this.error, counterEpochs);
            }

            if (this.error < 0.0001) {
                break;
            }

            this.runEpoch(dataset);
        }

        return this;
    }

    process(data) {
        let outputs: number[] = [];
        data = new Float64Array(data);

        console.log(data);

        this.layers.forEach((layer: Neuron[]) => {
            if (outputs.length > 0) {
                data = new Float64Array(outputs);
                outputs = [];
            }

            this.layers.synapticProcessor.setData(data);

            for (let index = 0; index < layer.length; index++) {
                const neuron = layer[index];
                outputs[index] = neuron.process();
            }
        });

        // return outputs;
        console.log(outputs);

        return outputs.map(output => {
            return Math.round(output);
        });
    }

    addLayer(numberNeurons: number) {
        this.layers.add(numberNeurons);

        return this;
    }

    /**
     * @param {{ layers: number[]; weights: number[][] }} model
     * @returns {this}
     */
    importModel(model: ModelType): this {
        model.layers.forEach(layer => {
            this.addLayer(layer);
        });

        model.weights.forEach((layerWeights, index) => {
            this.layers.get(index).forEach((neuron: Neuron, neuronIndex) => {
                neuron
                    .setWeights(new Float64Array(layerWeights[neuronIndex]))
                    .setBeforeWeights(neuron.weights.slice())
                    .setThreshold(model.thresholds[index][neuronIndex]);
            });
        });

        return this;
    }

    /**
     * @returns {ModelType}
     */
    exportModel(): ModelType {
        let model: ModelType = {
            layers: [],
            thresholds: [],
            weights: []
        };

        this.layers.forEach(layer => {
            model.layers.push(layer.length);

            const indexLayerThresholds = model.thresholds.push([]);
            const indexLayerWeights = model.weights.push([]);

            let layerThresholds = model.thresholds[indexLayerThresholds - 1];
            let layerWeights = model.weights[indexLayerWeights - 1];

            layer.forEach(neuron => {
                layerWeights.push(Array.from(neuron.weights));
                layerThresholds.push(neuron.threshold);
            });
        });

        return model;
    }

    /**
     * @param {Array<{ input: any; output: number }>} dataset
     */
    datasetInputToFloatArray(dataset: Array<{ input: any; output: number }>) {
        for (let index = 0; index < dataset.length; index++) {
            const data = dataset[index];
            data.input = new Float64Array(data.input);
        }
    }

    /**
     * @private
     * @param {Array<{ input: Float64Array; output: number }>} dataset
     */
    private runEpoch(dataset: Array<{ input: Float64Array; output: number }>) {
        const layerLastIndex = this.layers.length() - 1;
        let sumErrors = 0;

        for (let dataIdx = 0; dataIdx < dataset.length; dataIdx++) {
            const data = dataset[dataIdx];

            this.forwardpropagation(data);
            this.backpropagation(data.output);

            this.layers.forEach((layer, layerIdx) => {
                for (let neuronIdx = 0; neuronIdx < layer.length; neuronIdx++) {
                    const neuron = layer[neuronIdx];

                    if (layerIdx === layerLastIndex) {
                        sumErrors += neuron.error * neuron.error;
                    }

                    neuron.recalculateWeights(data.input);
                }
            });
        }

        sumErrors /= 2 * dataset.length;
        this.error = parseFloat(sumErrors.toFixed(8));
    }

    /**
     * @private
     * @param {*} { input, output }
     * @returns
     */
    private forwardpropagation({ input, output }) {
        let outputs = [];
        let data = new Float64Array(input);

        this.layers.forEach((layer: Neuron[]) => {
            if (outputs.length > 0) {
                data = new Float64Array(outputs);
                outputs = [];
            }

            this.layers.synapticProcessor.setData(data);

            for (let index = 0; index < layer.length; index++) {
                const neuron = layer[index];

                neuron.setData(data, output).learn();
                outputs[index] = neuron.output();
            }
        });

        return this;
    }

    /**
     * @private
     * @param {number} output
     */
    private backpropagation(output: number) {
        const lastLayer = this.layers.getLast();

        for (let index = 0; index < lastLayer.length; index++) {
            const neuron = lastLayer[index];

            neuron.calculateErrorOfOutput(output);
            neuron.backpropagation();
        }
    }
}
