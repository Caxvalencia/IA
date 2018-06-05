import { ActivationFunctionType } from './activation-functions/activation-function';
import { Perceptron } from './perceptron';

export class Neuron extends Perceptron {
    beforeWeights: Float32Array;
    error: number;
    inputNeurons: Neuron[];
    outputNeurons: Neuron[];

    /**
     * @construtor
     */
    constructor(
        activationFunction: ActivationFunctionType = ActivationFunctionType.SIGMOIDAL
    ) {
        super(activationFunction);

        this.error = 0;
        this.outputNeurons = [];
        this.inputNeurons = [];
    }

    learn() {
        if (!this.weights) {
            this.assignWeights();
            this.setBeforeWeights(this.weights.slice());
        }

        for (let i = 0; i < this.dataStack.length; i++) {
            this.synapticProcessor
                .setData(this.dataStack[i][0])
                .setOutputExpected(this.dataStack[i][1])
                .calculateSynapses(this.weights);
        }

        return this;
    }

    /**
     * Error on hidden layers
     */
    backpropagation() {
        for (let index = 0; index < this.inputNeurons.length; index++) {
            this.inputNeurons[index].calculateHiddenError(index);
        }

        if (this.inputNeurons.length > 0) {
            this.inputNeurons[0].backpropagation();
        }
    }

    recalculateWeights() {
        const delta = this.synapticProcessor.learningRate * this.error;
        const momentumFactor = 0.8;
        let deltaWeights;

        for (let i = 0; i < this.weights.length; i++) {
            deltaWeights =
                momentumFactor * (this.weights[i] - this.beforeWeights[i]);

            this.beforeWeights[i] = this.weights[i];

            this.weights[i] +=
                this.synapticProcessor.data[i] * delta + deltaWeights;
            this.weights[i] = parseFloat(this.weights[i].toFixed(4));
        }
    }

    output(): number {
        return this.synapticProcessor.output();
    }

    calculateErrorOfOutput() {
        this.synapticProcessor.calculateError();
        this.error = this.synapticProcessor.error;
        this.calculateErrorDerivated(this.error);
    }

    calculateHiddenError(neuronIndex) {
        let sumError = 0;

        for (let index = 0; index < this.outputNeurons.length; index++) {
            const neuron = this.outputNeurons[index];
            sumError += neuron.weights[neuronIndex] * neuron.error;
        }

        this.calculateErrorDerivated(sumError);

        return this;
    }

    calculateErrorDerivated(factorDelta) {
        this.error = factorDelta * this.prime();

        return this;
    }

    /**
     * @returns {number}
     */
    prime(): number {
        return this.synapticProcessor.activationFunction.prime(
            this.synapticProcessor.synapse
        );
    }

    /**
     * @param {Float32Array} beforeWeights
     * @returns {this}
     */
    setBeforeWeights(beforeWeights: Float32Array): this {
        this.beforeWeights = beforeWeights;

        return this;
    }

    /**
     * @returns {this}
     */
    initThreshold(): this {
        this.synapticProcessor.threshold = this.createWeight();

        return this;
    }
}
