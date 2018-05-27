import {
    ActivationFunction,
    ActivationFunctionType
} from './activation-functions/activation-function';

export class SynapticProcessor {
    activationFunction: string;
    error: number;
    synapse: number;
    data: Float32Array;
    threshold: number;
    learningRate: number;
    delta: number;
    private outputExpected: number;

    constructor(
        activationFunction: ActivationFunctionType,
        data: Float32Array = null,
        outputExpected: number = null,
        learningRate: number = 0.73
    ) {
        this.error = 0;
        this.activationFunction = activationFunction;

        this.setLearningFactor(learningRate);
        this.setOutputExpected(outputExpected);
        this.setData(data);
    }

    /**
     * @returns {number}
     */
    output(): number {
        return ActivationFunction.process(this.activationFunction)(
            this.synapse
        );
    }

    /**
     * @param {Float32Array} weights 
     */
    recalculateWeights(weights: Float32Array) {
        const error = this.outputExpected - this.output();

        this.calculateDelta(error);
        this.updateThreshold();

        for (let i = 0; i < weights.length; i++) {
            weights[i] += this.data[i] * this.delta;
        }
    }

    updateThreshold() {
        this.threshold += this.delta;
    }

    /**
     * @param {any} expectedOutput
     * @param {*} output
     * @returns this
     */
    calculateDelta(error) {
        this.delta = this.learningRate * error;

        return this;
    }

    /**
     * @param {Float32Array} weights 
     * @returns  
     */
    calculateSynapses(weights: Float32Array) {
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

    setData(data: Float32Array) {
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
