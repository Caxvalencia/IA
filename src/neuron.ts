import { SynapticProcessor } from './synaptic-processor';
import { ActivationFunctionType } from './activation-functions/activation-function';

export class Neuron {
    inputNeurons: Neuron[];
    outputNeurons: Neuron[];
    error: number;
    weights: number[];
    isHidden: boolean;
    activationFunction: ActivationFunctionType;
    rangeWeight: { MIN: number; MAX: number };
    synapticProcessor: SynapticProcessor;

    dataStack: any[];

    /**
     * @construtor
     */
    constructor(isHidden = false) {
        this.rangeWeight = { MIN: -5, MAX: 4.9 };
        this.isHidden = isHidden;

        this.weights = null;
        this.error = 0;

        this.outputNeurons = [];
        this.inputNeurons = [];

        this.activationFunction = ActivationFunctionType.SIGMOIDAL;
        this.synapticProcessor = new SynapticProcessor(this.activationFunction);
        this.dataStack = [];
    }

    addData(data: number[], output?) {
        this.dataStack.push([data, output]);

        return this;
    }

    learn() {
        if (!this.weights) {
            this.assignWeights();
        }

        for (let i = 0; i < this.dataStack.length; i++) {
            this.synapticProcessor
                .setData(this.dataStack[i][0])
                .setExpectedOutput(this.dataStack[i][1])
                .calculateSynapses(this.weights)
                .calculateErrorDerivated();
        }

        return this;
    }

    process(data: any[]) {
        let synapticProcessor = new SynapticProcessor(
            this.activationFunction,
            data
        );

        return synapticProcessor.calculateSynapses(this.weights).output();
    }

    /**
     * Metodos publicos
     */
    backpropagation() {
        // Error en las capas ocultas
        this.inputNeurons.forEach((neuron: Neuron) => {
            neuron.calculateHiddenError();
        });

        if (this.inputNeurons.length > 0) {
            this.inputNeurons[0].backpropagation();
        }
    }

    recalculateWeights() {
        if (this.synapticProcessor.error !== 0) {
            this.synapticProcessor.recalculateWeights(this.weights);
        }
    }

    output(): number {
        return this.synapticProcessor.output();
    }

    calculateHiddenError() {
        let sumError = 0;
        let output = this.output();

        this.outputNeurons.forEach((neuron: Neuron) => {
            neuron.weights.forEach((weight: number) => {
                sumError += neuron.synapticProcessor.error * weight;
            });
        });

        this.error = sumError * (1 - output) * output;

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

        return this;
    }

    setWeights(weights) {
        this.weights = weights;

        return this;
    }
}
