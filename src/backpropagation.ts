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
            activationFunction: ActivationFunctionType.SIGMOIDAL
        }
    ) {
        this.error = 0;
        this.activationFunction = config.activationFunction;
        this.layers = new Layer(this.activationFunction);
        this.epochs = config.epochs;
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

    forwardpropagation({ input, output }) {
        let outputs = [];
        let data = new Float64Array(input);

        this.layers.forEach((layer: Neuron[]) => {
            if (outputs.length > 0) {
                data = new Float64Array(outputs);
                outputs = [];
            }

            this.layers.synapticProcessor
                .setData(data)
                .setOutputExpected(output);

            for (let index = 0; index < layer.length; index++) {
                const neuron = layer[index];

                neuron.setData(data, output).learn();
                outputs[index] = neuron.output();
            }
        });

        return this;
    }

    backpropagation() {
        const lastLayer = this.layers.getLast();

        for (let index = 0; index < lastLayer.length; index++) {
            const neuron = lastLayer[index];

            neuron.calculateErrorOfOutput();
            neuron.backpropagation();
        }
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

    addLayer(numberNeurons: number) {
        this.layers.add(numberNeurons);

        return this;
    }

    process(data) {
        let outputs: number[] = [];

        console.log(data);

        this.layers.forEach(layer => {
            if (outputs.length > 0) {
                data = outputs;
                outputs = [];
            }

            for (let index = 0; index < layer.length; index++) {
                const neuron = layer[index];
                outputs[index] = neuron.process(data);
            }
        });

        // return outputs;
        console.log(outputs);

        return outputs.map(output => {
            return Math.round(output);
        });
    }

    exportModel() {
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

    setError(error) {
        this.error = error;

        return this;
    }

    datasetInputToFloatArray(dataset: Array<{ input: any; output: number }>) {
        for (let index = 0; index < dataset.length; index++) {
            const data = dataset[index];
            data.input = new Float64Array(data.input);
        }
    }

    private runEpoch(dataset: Array<{ input: Float64Array; output: number }>) {
        let sumErrors = 0;

        for (let dataIdx = 0; dataIdx < dataset.length; dataIdx++) {
            const data = dataset[dataIdx];

            this.forwardpropagation(data);
            this.backpropagation();

            this.layers.forEach(layer => {
                for (let layerIdx = 0; layerIdx < layer.length; layerIdx++) {
                    const neuron = layer[layerIdx];

                    sumErrors += neuron.error * neuron.error;
                    neuron.recalculateWeights(data.input);
                }
            });

            sumErrors /= 2;
        }

        this.setError(parseFloat(sumErrors.toFixed(8)));
    }
}
