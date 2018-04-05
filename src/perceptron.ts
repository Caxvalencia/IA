import { SynapticProcessor } from './synaptic-processor';

export const LIMIT_ERRORS: number = 10000;

export class Perceptron {
    counterErrors: number;
    synapticProcessor: SynapticProcessor[];
    hasError: boolean;
    weights: number[];
    rangeWeight: { MIN: number; MAX: number };

    private funcBack: () => void;
    private activationFunction: string;

    constructor() {
        this.rangeWeight = { MIN: -5, MAX: 4.9 };
        this.synapticProcessor = [];
        this.counterErrors = 0;
        this.hasError = false;
        this.weights = null;
        this.funcBack = () => {};
    }

    addData(data, output) {
        if (data[0] === undefined) {
            return this;
        }

        if (data[0][0] === undefined) {
            this.synapticProcessor.push(
                new SynapticProcessor(data, output, this.activationFunction)
            );

            return this;
        }

        for (let i = 0; i < data.length; i++) {
            this.synapticProcessor.push(
                new SynapticProcessor(
                    data[i],
                    output[i],
                    this.activationFunction
                )
            );
        }

        return this;
    }

    learn() {
        if (this.synapticProcessor.length === 0) {
            return;
        }

        if (!this.weights) {
            this.assignWeights();
        }

        let synapticProcessor: SynapticProcessor = null;

        this.hasError = false;

        for (let i = 0; i < this.synapticProcessor.length; i++) {
            synapticProcessor = this.synapticProcessor[i];

            synapticProcessor.calculateSynapses(this.weights);
            synapticProcessor.calculateError();

            if (synapticProcessor.error !== 0) {
                this.hasError = true;
                synapticProcessor.recalculateWeights(this.weights);
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

    process(data, activationFunction?) {
        let synapticProcessor = new SynapticProcessor(
            data,
            null,
            activationFunction
        );

        synapticProcessor.calculateSynapses(this.weights);

        return synapticProcessor.output();
    }

    setWeights(weights) {
        this.weights = weights;

        return this;
    }

    setActivationFunction(activationFunction: string) {
        this.activationFunction = activationFunction;

        return this;
    }

    private assignWeights() {
        let dataSize = this.synapticProcessor[0].data.length;
        let weights = new Array<number>(dataSize);
        let range = this.rangeWeight.MAX - this.rangeWeight.MIN;

        for (let i = 0; i < dataSize; i++) {
            while (!weights[i]) {
                weights[i] = parseFloat(
                    (Math.random() * range + this.rangeWeight.MIN).toFixed(4)
                );
            }
        }

        this.setWeights(weights);
        this.funcBack();
    }
}
