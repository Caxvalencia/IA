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

    /**
     * Error on hidden layers
     */
    backpropagation() {
        this.inputNeurons.forEach((neuron: Neuron, neuronIndex) => {
            neuron.calculateHiddenError(neuronIndex);
        });

        if (this.inputNeurons.length > 0) {
            this.inputNeurons[0].backpropagation();
        }
    }

    recalculateWeights() {
        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i] +=
                this.synapticProcessor.learningRate *
                this.synapticProcessor.data[i] *
                this.error;

            this.weights[i] = parseFloat(this.weights[i].toFixed(4));
        }
    }

    output(): number {
        return this.synapticProcessor.output();
    }

    calculateErrorOfOutput() {
        this.synapticProcessor.calculateError();
        this.calculateErrorDerivated(this.synapticProcessor.error);
    }

    calculateHiddenError(neuronIndex) {
        let sumError = 0;

        this.outputNeurons.forEach((neuron: Neuron) => {
            sumError += neuron.weights[neuronIndex] * neuron.error;
        });

        this.calculateErrorDerivated(sumError);

        return this;
    }

    calculateErrorDerivated(factorDelta) {
        const output = this.output();
        const errorComplement = output - output * output;

        this.error = factorDelta * errorComplement;

        return this;
    }
}
