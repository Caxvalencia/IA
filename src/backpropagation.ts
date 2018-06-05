import { ActivationFunctionType } from './activation-functions/activation-function';
import { Layer } from './layer';
import { Neuron } from './neuron';

interface BackpropagationConfig {
    epochs: number;
    activationFunction?: ActivationFunctionType;
}

'use strict';
export class Backpropagation {
    layers: Layer;
    LIMIT_ERRORS: number;
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
        this.LIMIT_ERRORS = config.epochs;
    }

    /**
     * @param {{ layers: number[]; weights: number[][] }} model
     * @returns {this}
     */
    importModel(model: { layers: number[]; weights: number[][][] }): this {
        model.layers.forEach(layer => {
            this.addLayer(layer);
        });

        model.weights.forEach((layerWeights, index) => {
            this.layers.get(index).forEach((neuron: Neuron, neuronIndex) => {
                neuron
                    .setWeights(new Float64Array(layerWeights[neuronIndex]))
                    .setBeforeWeights(neuron.weights.slice())
                    .initThreshold();
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

            for (let index = 0; index < layer.length; index++) {
                const neuron = layer[index];

                neuron.addData(data, output).learn();
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

    learn(data: Array<{ input: number[]; output: number }>) {
        this.runEpoch(data);
        let counterEpochs = 1;

        while (counterEpochs <= this.LIMIT_ERRORS) {
            counterEpochs++;

            if (counterEpochs % 1000 === 0) {
                console.log(this.error, counterEpochs);
            }

            if (this.error < 0.0001) {
                break;
            }

            this.runEpoch(data);
        }

        return this;
    }

    addLayer(numberNeurons: number) {
        this.layers.add(numberNeurons);

        return this;
    }

    process(data) {
        let outputs = [];

        console.log(data);

        this.layers.forEach(layer => {
            if (outputs.length > 0) {
                data = outputs;
                outputs = [];
            }

            for (let index = 0; index < layer.length; index++) {
                outputs.push(layer[index].process(data));
            }
        });

        // return outputs;
        console.log(outputs);

        return outputs.map(output => {
            return Math.round(output);
        });
    }

    setError(error) {
        this.error = error;

        return this;
    }

    private runEpoch(data: Array<{ input: number[]; output: number }>) {
        let sumErrors = 0;

        for (let jindex = 0; jindex < data.length; jindex++) {
            this.forwardpropagation(data[jindex]);
            this.backpropagation();

            this.layers.forEach(layer => {
                for (let index = 0; index < layer.length; index++) {
                    const neuron = layer[index];

                    sumErrors += neuron.error * neuron.error;
                    neuron.recalculateWeights();
                }
            });

            sumErrors /= 2;
        }

        this.setError(parseFloat(sumErrors.toFixed(8)));
    }
}
