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

    forwardpropagation({ input, output }) {
        let outputs = [];
        let data = input;

        this.layers.forEach((layer: Neuron[]) => {
            if (outputs.length > 0) {
                data = outputs;
                outputs = [];
            }

            layer.forEach((neuron: Neuron) => {
                neuron.addData(data, output).learn();
                outputs.push(neuron.output());
            });
        });

        return this;
    }

    backpropagation() {
        const lastLayer = this.layers.getLast();

        lastLayer.forEach((neuron: Neuron) => {
            neuron.calculateErrorOfOutput();
            neuron.backpropagation();
        });
    }

    learn(datas: Array<{ input: number[]; output: number }>) {
        let counterErrors = 0;

        this.runEpoch(datas);

        while (this.error > 0.001) {
            counterErrors++;

            if (counterErrors % 1000 === 0) {
                console.log(this.error, counterErrors);
            }

            if (counterErrors >= this.LIMIT_ERRORS) {
                return this;
            }

            this.runEpoch(datas);
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

            layer.forEach((neuron: Neuron) => {
                outputs.push(neuron.process(data));
            });
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

    private runEpoch(datas: Array<{ input: number[]; output: number }>) {
        let sumErrors = 0;

        datas.forEach(data => {
            this.forwardpropagation(data);
            this.backpropagation();

            this.layers.forEach(layer => {
                layer.forEach((neuron: Neuron) => {
                    sumErrors += neuron.error * neuron.error;
                    neuron.recalculateWeights();
                });
            });

            sumErrors /= 2;
        });

        this.setError(parseFloat(sumErrors.toFixed(4)));
    }
}
