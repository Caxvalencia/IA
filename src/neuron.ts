import { SynapticProcessor } from './synaptic-processor';

export class Neuron {
    activationFunction: string;
    threshold: number;
    inputNeurons: any[];
    outputNeurons: any[];
    error: number;
    synapse: number;
    weights: any;
    learningFactor: number;
    isHidden: boolean;
    rangeWeight: { MIN: number; MAX: number };
    synapticProcessor: SynapticProcessor[];

    /**
     * @construtor
     */
    constructor(isHidden = false) {
        this.rangeWeight = { MIN: -5, MAX: 4.9 };
        this.synapticProcessor = [];
        this.learningFactor = 0.5;
        this.isHidden = isHidden;

        this.weights = null;
        this.synapse = 0;
        this.error = 0;

        this.outputNeurons = [];
        this.inputNeurons = [];

        this.threshold = 1;
        this.activationFunction = 'sigmoidal';
    }

    addData(data: any[], output?) {
        if (data[0][0] === undefined) {
            data = [data];
            output = [output];
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

        this.synapticProcessor.forEach(synapticProcessor => {
            synapticProcessor.calculateSynapses(this.weights);
            synapticProcessor.calculateErrorDerivated();
        });

        return this;
    }

    process(data: any[]) {
        let synapticProcessor = new SynapticProcessor(
            data,
            null,
            this.activationFunction
        );
        synapticProcessor.calculateSynapses(this.weights);

        return synapticProcessor.output();
    }

    /**
     * Metodos publicos
     */
    backpropagation() {
        // Error en las capas ocultas
        this.inputNeurons.forEach((neurona: Neuron, idx) => {
            neurona.calculateHiddenError(idx);
        });

        if (this.inputNeurons.length > 0) {
            this.inputNeurons[0].backpropagation();
        }
    }

    recalculateWeights() {
        this.synapticProcessor.forEach(synapticProcessor => {
            if (synapticProcessor.error !== 0) {
                synapticProcessor.recalculateWeights(this.weights);
            }
        });
    }

    output() {
        let output = [];

        this.synapticProcessor.forEach(synapticProcessor => {
            output.push(synapticProcessor.output());
        });

        return output;
    }

    calculateHiddenError(idx) {
        let output = this.output();
        let sumError = 0;

        this.outputNeurons.forEach((neurona: Neuron) => {
            sumError += neurona.error * neurona.weights[idx];
        });

        this.error = sumError * (1 - output[idx]) * output[idx];

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
        let dataSize = this.synapticProcessor[0].data.length;
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

    setLearningFactor(learningFactor) {
        this.learningFactor = learningFactor;

        return this;
    }
}
