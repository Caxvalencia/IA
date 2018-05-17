import {
    ActivationFunction,
    ActivationFunctionType
} from './activation-functions/activation-function';

export class SynapticProcessor {
    activationFunction: string;
    error: number;
    synapse: number;
    expectedOutput: any;
    data: number[];
    threshold: number;
    learningRate: number;
    delta: number;

    constructor(
        activationFunction: ActivationFunctionType,
        data: any[] = null,
        expectedOutput: any = null
    ) {
        this.learningRate = 0.25;
        this.error = 0;
        this.activationFunction = activationFunction;

        this.setExpectedOutput(expectedOutput);
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
     * @param {number[]} weights
     */
    recalculateWeights(weights: number[]) {
        let error = this.expectedOutput - this.output();

        this.updateThreshold(error);
        this.calculateDelta(error);

        for (let i = 0; i < weights.length; i++) {
            weights[i] += this.data[i] * this.delta;
        }
    }

    updateThreshold(error): any {
        this.threshold -= this.learningRate * error;
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
     * @param {number[]} weights
     * @returns
     */
    calculateSynapses(weights: number[]) {
        this.synapse = 0;

        for (let i = 0; i < weights.length; i++) {
            this.synapse += this.data[i] * weights[i];
        }

        this.synapse -= this.threshold;

        return this;
    }

    calculateError() {
        this.error = this.expectedOutput - this.output();

        return this;
    }

    calculateErrorDerivated(factorDelta) {
        let output = this.output();
        let errorComplement = output - output * output;

        this.delta = factorDelta * errorComplement;

        return this;
    }

    setData(data: any[]) {
        if (data === null) {
            return this;
        }

        this.data = data.slice();

        return this;
    }

    setExpectedOutput(expectedOutput) {
        this.expectedOutput = expectedOutput;

        return this;
    }

    setLearningFactor(learningFactor) {
        this.learningRate = learningFactor;

        return this;
    }
}
