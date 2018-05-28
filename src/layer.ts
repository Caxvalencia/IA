import { Neuron } from './neuron';

export class Layer {
    private layers: Neuron[][];

    constructor() {
        this.layers = [];
    }

    /**
     * @param {number} numberNeurons
     * @returns
     */
    add(numberNeurons: number) {
        const layer = this.create(numberNeurons);
        const indexNewLayer = this.layers.push(layer) - 1;
        const beforeLayer = this.layers[indexNewLayer - 1];

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

    forEach(callback) {
        return this.layers.forEach(callback);
    }

    /**
     * @returns {Neuron[]}
     */
    getLast(): Neuron[] {
        return this.layers[this.layers.length - 1];
    }

    /**
     * @private
     * @param {number} numberNeurons
     * @returns {Neuron[]}
     */
    private create(numberNeurons: number): Neuron[] {
        const layer: Neuron[] = [];
        const isHidden: boolean = this.layers.length > 0;

        for (let i = 0; i < numberNeurons; i++) {
            layer[i] = new Neuron(isHidden);
        }

        return layer;
    }
}
