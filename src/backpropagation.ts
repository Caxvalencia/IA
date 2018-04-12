import { Neuron } from './neuron';

'use strict';
export class Backpropagation {
    LIMIT_ERRORS: number;
    counterErrors: number;
    funcionActivacion: string;
    error: number;
    factorAprendizaje: number;
    layers: Neuron[][];

    /**
     * @construtor
     */
    constructor() {
        this.layers = [];
        this.factorAprendizaje = 0.25;
        this.error = 0;
        this.funcionActivacion = 'sigmoidal';

        this.counterErrors = 0;
        this.LIMIT_ERRORS = 100000;
    }

    forwardPropagationData(data: { input: any; output: any }) {
        let outputs = [];
        let values = data.input;

        this.layers.forEach((layer: Neuron[]) => {
            if (outputs.length > 0) {
                values = outputs;
                outputs = [];
            }

            layer.forEach((neuron: Neuron) => {
                neuron.addData(values, data.output).learn();

                if (neuron.outputNeurons.length > 0) {
                    outputs.push(neuron.output());
                }
            });
        });

        return this;
    }

    learn(datas) {
        let learnCallback = () => {
            const indexLastLayer = this.layers.length - 1;
            let sumErrors = 0;

            datas.forEach(data => {
                // Forwardpropagation
                this.forwardPropagationData(data);

                this.layers[indexLastLayer].forEach((neuron: Neuron) => {
                    //Solo con una neurona tenemos acceso a todas las otras
                    neuron.backpropagation();
                });

                this.layers.forEach(layer => {
                    layer.forEach((neuron: Neuron) => {
                        sumErrors += Math.pow(neuron.error, 2);
                        neuron.recalculateWeights();
                    });
                });

                sumErrors *= 0.5;
            });

            this.setError(sumErrors);
        };

        learnCallback();

        while (this.error > 0.0001) {
            this.counterErrors++;

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

    addNeuron(layer, position?) {
        let neuron = new Neuron();

        if (!position) {
            this.layers[layer].push(neuron);

            return;
        }

        this.layers[layer][position] = neuron;
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
