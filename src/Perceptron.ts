import { ProcesadorSinaptico } from './ProcesadorSinaptico';

export const LIMIT_ERRORS: number = 10000;

export class Perceptron {
    counterErrors: number;
    procesadorSinaptico: ProcesadorSinaptico[];
    hasError: boolean;
    weight: any;
    rangoPesos: { MIN: number; MAX: number };

    private funcBack: () => void;
    private funcionActivacion: any;

    constructor() {
        this.rangoPesos = { MIN: -5, MAX: 4.9 };
        this.hasError = false;
        this.procesadorSinaptico = [];

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
            this.procesadorSinaptico.push(
                new ProcesadorSinaptico(datos, salida, this.funcionActivacion)
            );

            return this;
        }

        for (let i = 0; i < datos.length; i++) {
            this.procesadorSinaptico.push(
                new ProcesadorSinaptico(
                    datos[i],
                    salida[i],
                    this.funcionActivacion
                )
            );
        }

        return this;
    }

    aprender() {
        if (this.procesadorSinaptico.length === 0) {
            return;
        }

        if (!this.weight) {
            this.assignWeights();
        }

        let procesadorSinaptico = null;
        let len = this.procesadorSinaptico.length;

        this.hasError = false;

        for (let i = 0; i < len; i++) {
            procesadorSinaptico = this.procesadorSinaptico[i];

            procesadorSinaptico.calcularSinapsis(this.weight);
            procesadorSinaptico.calcularError();

            if (procesadorSinaptico.error !== 0) {
                this.hasError = true;
                procesadorSinaptico.reajustarPesos(this.weight);
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
        let procesadorSinaptico = new ProcesadorSinaptico(
            datos,
            null,
            funcionActivacion
        );

        procesadorSinaptico.calcularSinapsis(this.weight);

        return procesadorSinaptico.salida();
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
        let len = this.procesadorSinaptico[0].datos.length;
        let pesos = new Array(len);
        let rango = this.rangoPesos.MAX - this.rangoPesos.MIN;

        for (let i = 0; i < len; i++) {
            while (!pesos[i]) {
                pesos[i] = parseFloat(
                    (Math.random() * rango + this.rangoPesos.MIN).toFixed(4)
                );
            }
        }

        this.setWeight(pesos);
        this.funcBack();
    }
}
