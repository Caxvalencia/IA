import {
    ActivationFunction,
    ActivationFunctionType
} from './activation-functions/activation-function';

export class SynapticProcessor {
    activationFunction: ActivationFunction;
    error: number;
    synapse: number;
    data: Float64Array;
    threshold: number;
    learningRate: number;
    delta: number;
    private outputExpected: number;

    constructor(
        activationFunction: ActivationFunctionType,
        data: Float64Array = null,
        outputExpected: number = null,
        learningRate: number = 0.3
    ) {
        this.error = 0;
        this.activationFunction = ActivationFunction.init(activationFunction);

        this.setLearningFactor(learningRate);
        this.setOutputExpected(outputExpected);
        this.setData(data);
    }

    /**
     * @returns {number}
     */
    output(): number {
        return this.activationFunction.process(this.synapse);
    }

    /**
     * @param {Float64Array} weights
     */
    recalculateWeights(weights: Float64Array) {
        const error = this.outputExpected - this.output();
        this.delta = this.learningRate * error;
        this.threshold += this.delta;

        for (let i = 0; i < weights.length; i++) {
            weights[i] += this.data[i] * this.delta;
        }
    }

    /**
     * @param {Float64Array} weights
     * @returns
     */
    calculateSynapses(weights: Float64Array) {
        this.synapse = 0;

        for (let i = 0; i < weights.length; i++) {
            this.synapse += this.data[i] * weights[i];
        }

        this.synapse += this.threshold;

        return this;
    }

    calculateError() {
        this.error = this.outputExpected - this.output();

        return this;
    }

    setData(data: Float64Array) {
        if (data === null) {
            return this;
        }

        this.data = data.slice();

        return this;
    }

    setOutputExpected(expectedOutput) {
        this.outputExpected = expectedOutput;

        return this;
    }

    setLearningFactor(learningFactor) {
        this.learningRate = learningFactor;

        return this;
    }
}
