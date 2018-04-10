import { SynapticProcessor } from './synaptic-processor';

export class Neuron {
    data: any[];
    threshold: number;
    inputNeurons: any[];
    outputNeurons: any[];
    error: number;
    synapse: number;
    weights: any;
    learningFactor: number;
    isHidden: boolean;
    rangeWeight: { MIN: number; MAX: number };

    /**
     * @construtor
     */
    constructor(isHidden = false) {
        this.rangeWeight = { MIN: -5, MAX: 4.9 };
        this.learningFactor = 0.5;
        this.isHidden = isHidden;

        this.weights = null;
        this.synapse = 0;
        this.error = 0;

        this.outputNeurons = [];
        this.inputNeurons = [];

        this.threshold = 1;
        this.data = [];
    }

    process(data: any[]) {
        let synapticProcessor = new SynapticProcessor(data, null, 'sigmoidal');
        synapticProcessor.calculateSynapses(this.weights);

        return synapticProcessor.output();
    }

    calculateSynapses() {
        this.synapse = 0;

        for (let i = 0; i < this.weights.length; i++) {
            this.synapse += this.data[i] * this.weights[i];
        }

        return this;
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

    reajustarPesos() {
        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i] += this.learningFactor * this.error * this.data[i];
        }
    }

    output() {
        //Funcion de activacion sigmoidal binaria
        return 1 / (1 + Math.pow(Math.E, -this.synapse));
    }

    calculateError(expectedOutput) {
        let output = this.output();
        this.error = (expectedOutput - output) * (1 - output) * output;

        return this;
    }

    calculateHiddenError(idx) {
        let output = this.output();
        let sumError = 0;

        this.outputNeurons.forEach((neurona: Neuron) => {
            sumError += neurona.error * neurona.weights[idx];
        });

        this.error = sumError * (1 - output) * output;

        return this;
    }

    setLearningFactor(learningFactor) {
        this.learningFactor = learningFactor;

        return this;
    }

    setData(data: any[]) {
        data.push(this.threshold);
        this.data = data;

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

    public assignWeights() {
        let dataSize = this.data.length;
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
