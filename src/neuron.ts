import { ActivationFunctionType } from './activation-functions/activation-function';
import { Perceptron } from './perceptron';

export class Neuron extends Perceptron {
    inputNeurons: Neuron[];
    outputNeurons: Neuron[];

    error: number;
    isHidden: boolean;

    /**
     * @construtor
     */
    constructor(isHidden = false) {
        super(null, ActivationFunctionType.SIGMOIDAL);

        this.isHidden = isHidden;
        this.error = 0;

        this.outputNeurons = [];
        this.inputNeurons = [];
    }

    learn() {
        if (!this.weights) {
            this.assignWeights();
        }

        for (let i = 0; i < this.dataStack.length; i++) {
            this.synapticProcessor
                .setData(this.dataStack[i][0])
                .setExpectedOutput(this.dataStack[i][1])
                .calculateSynapses(this.weights);
        }

        return this;
    }

    backpropagation() {
        // Error en las capas ocultas
        this.inputNeurons.forEach((neuron: Neuron, neuronIndex) => {
            neuron.calculateHiddenError(neuronIndex);
        });

        if (this.inputNeurons.length > 0) {
            this.inputNeurons[0].backpropagation();
        }
    }

    recalculateWeights() {
        if (this.synapticProcessor.error !== 0) {
            this.synapticProcessor.recalculateWeights(this.weights);
        }
    }

    output(): number {
        return this.synapticProcessor.output();
    }

    calculateHiddenError(neuronIndex) {
        let sumError = 0;

        this.outputNeurons.forEach((neuron: Neuron) => {
            sumError +=
                neuron.weights[neuronIndex] * neuron.synapticProcessor.delta;
        });

        this.synapticProcessor.calculateErrorDerivated(sumError);
        this.recalculateWeights();

        return this;
    }
}
