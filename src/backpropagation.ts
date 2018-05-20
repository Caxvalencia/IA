import { Neuron } from './neuron';

'use strict';
export class Backpropagation {
    LIMIT_ERRORS: number;
    counterErrors: number;
    funcionActivacion: string;
    error: number;
    layers: Neuron[][];

    /**
     * @construtor
     */
    constructor() {
        this.layers = [];
        this.error = 0;
        this.funcionActivacion = 'sigmoidal';

        this.counterErrors = 0;
        this.LIMIT_ERRORS = 2500;
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
        const lastLayer = this.layers[this.layers.length - 1];

        lastLayer.forEach((neuron: Neuron) => {
            neuron.synapticProcessor.calculateError();
            neuron.error = Math.abs(neuron.synapticProcessor.error);
            neuron.synapticProcessor.calculateErrorDerivated(neuron.error);

            if (neuron.error >= 0.0001) {
                neuron.recalculateWeights();
            }

            neuron.backpropagation();
        });
    }

    learn(datas: Array<{ input: number[]; output: number }>) {
        let learnCallback = () => {
            let sumErrors = 0;

            datas.forEach(data => {
                this.forwardpropagation(data);
                this.backpropagation();

                this.layers.forEach(layer => {
                    layer.forEach((neuron: Neuron) => {
                        sumErrors += neuron.error * neuron.error;
                    });
                });

                sumErrors /= 2;
            });

            // if (sumErrors >= this.error) {
            // this.layers.forEach(layer => {
            //     layer.forEach((neuron: Neuron) => {
            //         neuron.synapticProcessor.learningRate -= 0.01;
            //     });
            // });
            // }

            this.setError(parseFloat(sumErrors.toFixed(4)));
        };

        learnCallback();

        while (this.error > 0.001) {
            this.counterErrors++;

            console.log(this.error, this.counterErrors);

            if (this.counterErrors >= this.LIMIT_ERRORS) {
                // this.counterErrors = 0;
                return this;
            }

            learnCallback();
        }

        this.counterErrors = 0;

        return this;
    }

    addLayer(numberNeurons: number) {
        let layer = this.createLayer(numberNeurons);
        let indexNewLayer = this.layers.push(layer) - 1;
        let beforeLayer = this.layers[indexNewLayer - 1];

        // Verificar si existe capa anterior
        if (beforeLayer === undefined) {
            return this;
        }

        // Apuntar con cada Neuron de la nueva capa a la anterior
        layer.forEach((neuron: Neuron) => {
            neuron.inputNeurons = beforeLayer;
        });

        // Apuntar con cada neurona de la capa anterior a la nueva capa
        beforeLayer.forEach((neuron: Neuron) => {
            neuron.outputNeurons = layer;
        });

        return this;
    }

    process(data) {
        let outputs = [];

        this.layers.forEach(layer => {
            if (outputs.length > 0) {
                data = outputs;
                outputs = [];
            }

            layer.forEach((neuron: Neuron) => {
                outputs.push(neuron.process(data));
            });
        });

        return outputs.map(output => {
            return Math.round(output);
        });
    }

    setError(error) {
        this.error = error;

        return this;
    }

    private createLayer(numberNeurons: number) {
        let layer: Neuron[] = [];
        let isHidden: boolean = this.layers.length > 0;

        for (let i = 0; i < numberNeurons; i++) {
            layer[i] = new Neuron(isHidden);
        }

        return layer;
    }
}
