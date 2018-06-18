import { ActivationFunctionType } from './activation-functions/activation-function';
import { Perceptron } from './perceptron';

export class Neuron extends Perceptron {
    beforeWeights: Float64Array;
    error: number;
    inputNeurons: Neuron[];
    outputNeurons: Neuron[];
    synapse: number;

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

        this.synapticProcessor.calculateSynapses(this.weights, this.threshold);
        this.synapse = this.synapticProcessor.synapse;

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

    recalculateWeights(data: Float64Array) {
        const delta = /*this.synapticProcessor.learningRate*/ 0.73 * this.error;
        const momentumFactor = 0.8;
        let deltaWeights: number;

        for (let i = 0; i < this.weights.length; i++) {
            deltaWeights =
                momentumFactor * (this.weights[i] - this.beforeWeights[i]);

            this.beforeWeights[i] = this.weights[i];

            this.weights[i] += data[i] * delta + deltaWeights;
            this.weights[i] = parseFloat(this.weights[i].toFixed(4));
        }
    }

    output(): number {
        return this.synapticProcessor.activationFunction.process(this.synapse);
    }

    process(data: Float64Array) {
        this.synapticProcessor
            .setData(data)
            .calculateSynapses(this.weights, this.threshold);
        this.synapse = this.synapticProcessor.synapse;

        return this.output();
    }

    calculateErrorOfOutput() {
        this.calculateError();
        this.calculateErrorDerivated(this.error);
    }

    calculateError() {
        this.error = this.synapticProcessor.outputExpected - this.output();

        return this;
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
        return this.synapticProcessor.activationFunction.prime(this.synapse);
    }

    /**
     * @param {Float64Array} beforeWeights
     * @returns {this}
     */
    setBeforeWeights(beforeWeights: Float64Array): this {
        this.beforeWeights = beforeWeights;

        return this;
    }

    /**
     * @returns {this}
     */
    setThreshold(threshold: number = this.createWeight()): this {
        this.threshold = threshold;

        return this;
    }

    setData(data: Float64Array, output: number) {
        this.dataStack = [[data, output]];

        return this;
    }
}
