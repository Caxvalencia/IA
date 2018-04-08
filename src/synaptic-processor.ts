import { sigmoidal } from './activation-functions/sigmoidal.function';

export class SynapticProcessor {
    activationFunction: string;
    error: number;
    synapse: number;
    expectedOutput: any;
    data: number[];
    threshold: number;
    learningFactor: number;

    constructor(data: any[], expectedOutput, activationFunction: string) {
        this.learningFactor = 0.5;
        this.threshold = 1;

        this.synapse = 0;
        this.error = 0;
        this.activationFunction = activationFunction;

        this.setExpectedOutput(expectedOutput);
        this.setData(data);
    }

    /**
     * @returns {number}
     */
    output(): number {
        if (this.activationFunction === 'sigmoidal') {
            return sigmoidal(this.synapse);
        }

        return this.synapse >= 0 ? 1 : 0;
    }

    /**
     * @param {number[]} weights
     */
    recalculateWeights(weights: number[]) {
        for (let i = 0; i < weights.length; i++) {
            weights[i] += this.learningFactor * this.error * this.data[i];
        }
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

        return this;
    }

    calculateError() {
        this.error = this.expectedOutput - this.output();

        return this;
    }

    setData(data: any[]) {
        data.push(this.threshold);
        this.data = data;

        return this;
    }

    setExpectedOutput(expectedOutput) {
        this.expectedOutput = expectedOutput;

        return this;
    }

    setLearningFactor(learningFactor) {
        this.learningFactor = learningFactor;

        return this;
    }
}
