import { sigmoidal } from './activation-functions/sigmoidal.function';
export class SynapticProcessor {
    activationFunction: string;
    error: number;
    synapse: number;
    expectedOutput: any;
    data: number[];
    threshold: number;
    learningFactor: number;

    constructor(data, expectedOutput, activationFunction) {
        this.learningFactor = 0.5;
        this.threshold = 1;

        this.expectedOutput = expectedOutput;
        this.synapse = 0;
        this.error = 0;
        this.activationFunction = activationFunction;

        this.setData(data);
    }

    output() {
        if (this.activationFunction === 'sigmoidal') {
            return sigmoidal(this.synapse);
        }

        return this.synapse >= 0 ? 1 : 0;
    }

    recalculateWeights(weight) {
        for (let i = 0; i < weight.length; i++) {
            weight[i] += this.learningFactor * this.error * this.data[i];
        }
    }

    calculateSynapses(weight) {
        this.synapse = 0;

        for (let i = 0; i < weight.length; i++) {
            this.synapse += this.data[i] * weight[i];
        }

        return this;
    }

    calculateError() {
        this.error = this.expectedOutput - this.output();

        return this;
    }

    setData(data) {
        this.data = data ? [this.threshold].concat(data) : [this.threshold];

        return this;
    }

    setExpectedOutput(expectedOutput) {
        this.expectedOutput = expectedOutput;

        return this;
    }
}
