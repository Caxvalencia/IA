import { SynapticProcessor } from './SynapticProcessor';

export const LIMIT_ERRORS: number = 10000;

export class Perceptron {
    counterErrors: number;
    synapticProcessor: SynapticProcessor[];
    hasError: boolean;
    weight: any;
    rangeWeight: { MIN: number; MAX: number };

    private funcBack: () => void;
    private funcionActivacion: any;

    constructor() {
        this.rangeWeight = { MIN: -5, MAX: 4.9 };
        this.hasError = false;
        this.synapticProcessor = [];

        this.counterErrors = 0;

        this.weight = null;
        this.funcionActivacion = null;
        this.funcBack = () => {};
    }

    addDatos(datos, salida) {
        if (datos[0] === undefined) {
            return this;
        }

        if (datos[0][0] === undefined) {
            this.synapticProcessor.push(
                new SynapticProcessor(datos, salida, this.funcionActivacion)
            );

            return this;
        }

        for (let i = 0; i < datos.length; i++) {
            this.synapticProcessor.push(
                new SynapticProcessor(
                    datos[i],
                    salida[i],
                    this.funcionActivacion
                )
            );
        }

        return this;
    }

    aprender() {
        if (this.synapticProcessor.length === 0) {
            return;
        }

        if (!this.weight) {
            this.assignWeights();
        }

        let synapticProcessor = null;
        let len = this.synapticProcessor.length;

        this.hasError = false;

        for (let i = 0; i < len; i++) {
            synapticProcessor = this.synapticProcessor[i];

            synapticProcessor.calcularSinapsis(this.weight);
            synapticProcessor.calcularError();

            if (synapticProcessor.error !== 0) {
                this.hasError = true;
                synapticProcessor.reajustarPesos(this.weight);
                this.funcBack();
            }
        }

        if (this.hasError) {
            this.counterErrors++;

            if (this.counterErrors >= LIMIT_ERRORS) {
                this.counterErrors = 0;

                throw Error('Maximum error limit reached');
            }

            return this.aprender();
        }

        this.counterErrors = 0;

        return this;
    }

    procesar(datos, funcionActivacion?) {
        let synapticProcessor = new SynapticProcessor(
            datos,
            null,
            funcionActivacion
        );

        synapticProcessor.calcularSinapsis(this.weight);

        return synapticProcessor.salida();
    }

    getWeight() {
        return this.weight;
    }

    setWeight(weight) {
        this.weight = weight;

        return this;
    }

    setFuncionActivacion(funcionActivacion) {
        this.funcionActivacion = funcionActivacion;

        return this;
    }

    private assignWeights() {
        let len = this.synapticProcessor[0].data.length;
        let pesos = new Array(len);
        let rango = this.rangeWeight.MAX - this.rangeWeight.MIN;

        for (let i = 0; i < len; i++) {
            while (!pesos[i]) {
                pesos[i] = parseFloat(
                    (Math.random() * rango + this.rangeWeight.MIN).toFixed(4)
                );
            }
        }

        this.setWeight(pesos);
        this.funcBack();
    }
}