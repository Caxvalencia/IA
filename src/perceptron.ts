import { ActivationFunctionType } from './activation-functions/activation-function';
import { SynapticProcessor } from './synaptic-processor';

const LIMIT_ERRORS: number = 8000;

export class Perceptron {
    dataStack: any[];
    weights: Float64Array;
    counterErrors: number;
    hasError: boolean;

    synapticProcessor: SynapticProcessor;

    funcBack: () => void;

    constructor(
        activationFunction?: ActivationFunctionType,
        callback = () => {}
    ) {
        this.counterErrors = 0;
        this.hasError = false;
        this.weights = null;
        this.funcBack = callback;

        this.synapticProcessor = new SynapticProcessor(activationFunction);
        this.dataStack = [];
    }

    addData(data: Float64Array, output: number) {
        this.dataStack.push([data, output]);

        return this;
    }

    learn() {
        if (!this.weights) {
            this.assignWeights();
        }

        this.hasError = false;

        for (let i = 0; i < this.dataStack.length; i++) {
            this.synapticProcessor
                .setData(this.dataStack[i][0])
                .setOutputExpected(this.dataStack[i][1])
                .calculateSynapses(this.weights)
                .calculateError();

            if (this.synapticProcessor.error !== 0) {
                this.synapticProcessor.recalculateWeights(this.weights);

                this.hasError = true;
                this.funcBack();
            }
        }

        if (this.hasError) {
            this.counterErrors++;

            if (this.counterErrors >= LIMIT_ERRORS) {
                this.counterErrors = 0;

                throw Error('Maximum error limit reached');
            }

            return this.learn();
        }

        this.counterErrors = 0;

        return this;
    }

    process(data: Float64Array) {
        return this.synapticProcessor
            .setData(data)
            .calculateSynapses(this.weights)
            .output();
    }

    setWeights(weights: Float64Array) {
        this.weights = weights;

        return this;
    }

    protected createWeight() {
        const rangeWeight = { MIN: -0.5, MAX: 0.49 };
        const rangeDiff = rangeWeight.MAX - rangeWeight.MIN;
        let weight = 0;

        while (!weight) {
            weight = parseFloat(
                (Math.random() * rangeDiff + rangeWeight.MIN).toFixed(4)
            );
        }

        return weight;
    }

    protected assignWeights() {
        const dataSize: number = this.dataStack[0][0].length;
        const weights = new Float64Array(dataSize);

        for (let i = 0; i < dataSize; i++) {
            weights[i] = this.createWeight();
        }

        this.setWeights(weights);
        this.synapticProcessor.threshold = this.createWeight();

        this.funcBack();

        return this;
    }
}
