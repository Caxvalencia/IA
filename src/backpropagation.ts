import { Neuron } from './neuron';
import { Layer } from './layer';

'use strict';
export class Backpropagation {
    layers: Layer;
    LIMIT_ERRORS: number;
    counterErrors: number;
    funcionActivacion: string;
    error: number;

    /**
     * @construtor
     */
    constructor() {
        this.layers = new Layer();
        this.error = 0;
        this.funcionActivacion = 'sigmoidal';

        this.counterErrors = 0;
        this.LIMIT_ERRORS = 5000;
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
        this.runEpoch(datas);

        while (this.error > 0.001) {
            this.counterErrors++;

            if (this.counterErrors >= this.LIMIT_ERRORS) {
                return this;
            }

            this.runEpoch(datas);
        }

        this.counterErrors = 0;

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

        if (this.counterErrors % 1000 === 0) {
            console.log(this.error, this.counterErrors);
        }
    }
}
