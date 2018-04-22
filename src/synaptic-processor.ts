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
    learningFactor: number;
    delta: number;

    constructor(
        activationFunction: ActivationFunctionType,
        data: any[] = null,
        expectedOutput: any = null
    ) {
        this.learningFactor = 0.25;
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
        return ActivationFunction.process(this.activationFunction)(
            this.synapse
        );
    }

    /**
     * @param {number[]} weights
     */
    recalculateWeights(weights: number[]) {
        this.calculateDelta(this.expectedOutput, this.output());

        for (let i = 0; i < weights.length; i++) {
            weights[i] += this.data[i] * this.delta;
        }
    }

    /**
     * @param {any} expectedOutput 
     * @param {*} output 
     * @returns this
     */
    calculateDelta(expectedOutput, output: any) {
        let error = expectedOutput - output;
        this.delta = this.learningFactor * error;

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

        return this;
    }

    calculateError() {
        this.error = this.expectedOutput - this.output();

        return this;
    }

    calculateErrorDerivated() {
        let output = this.output();
        let errorCommitted = 1 - output;

        this.error = (this.expectedOutput - output) * errorCommitted * output;

        return this;
    }

    setData(data: any[]) {
        if (data === null) {
            return this;
        }

        this.data = data.slice();
        this.data.push(this.threshold);

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
