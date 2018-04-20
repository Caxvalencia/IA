import { ActivationFunctionType } from './activation-functions/activation-function';
import { SynapticProcessor } from './synaptic-processor';

export const LIMIT_ERRORS: number = 10000;

export class Perceptron {
    dataStack: any[];

    counterErrors: number;
    hasError: boolean;
    weights: number[];
    rangeWeight: { MIN: number; MAX: number };

    public funcBack: () => void;
    private activationFunction: ActivationFunctionType;

    constructor(callback?) {
        this.rangeWeight = { MIN: -5, MAX: 4.9 };
        this.counterErrors = 0;
        this.hasError = false;
        this.weights = null;
        this.funcBack = callback || (() => {});

        this.dataStack = [];
    }

    addData(data: any[], output) {
        this.dataStack.push([data, output]);

        return this;
    }

    learn() {
        if (!this.weights) {
            this.assignWeights();
        }

        this.hasError = false;

        let synapticProcessor = new SynapticProcessor(this.activationFunction);

        for (let i = 0; i < this.dataStack.length; i++) {
            synapticProcessor
                .setData(this.dataStack[i][0])
                .setExpectedOutput(this.dataStack[i][1])
                .calculateSynapses(this.weights)
                .calculateError();

            if (synapticProcessor.error !== 0) {
                synapticProcessor.recalculateWeights(this.weights);

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

    process(data) {
        let synapticProcessor = new SynapticProcessor(
            this.activationFunction,
            data
        );

        return synapticProcessor.calculateSynapses(this.weights).output();
    }

    setWeights(weights) {
        this.weights = weights;

        return this;
    }

    setActivationFunction(activationFunction: ActivationFunctionType) {
        this.activationFunction = activationFunction;

        return this;
    }

    private createWeight() {
        let weight = 0;
        let range = this.rangeWeight.MAX - this.rangeWeight.MIN;

        while (!weight) {
            weight = parseFloat(
                (Math.random() * range + this.rangeWeight.MIN).toFixed(4)
            );
        }

        return weight;
    }

    private assignWeights() {
        let dataSize = this.dataStack[0][0].length + 1;
        let weights = new Array<number>(dataSize);

        for (let i = 0; i < dataSize; i++) {
            weights[i] = this.createWeight();
        }

        this.setWeights(weights);
        this.funcBack();
    }
}
