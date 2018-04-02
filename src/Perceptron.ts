import { ProcesadorSinaptico } from './ProcesadorSinaptico';

export class Perceptron {
    LIMIT_ERRORS: number;
    counterErrors: number;
    procesadorSinaptico: ProcesadorSinaptico[];
    hasError: boolean;
    pesos: any;
    rangoPesos: { MIN: number; MAX: number };

    private funcBack: () => void;
    private funcionActivacion: any;

    constructor() {
        this.rangoPesos = { MIN: -5, MAX: 4.9 };
        this.hasError = false;
        this.procesadorSinaptico = [];

        this.counterErrors = 0;
        this.LIMIT_ERRORS = 10000;

        this.pesos = null;
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

        if (!this.pesos) {
            this.asignarPesos();
        }

        let procesadorSinaptico = null;
        let len = this.procesadorSinaptico.length;

        this.hasError = false;

        for (let i = 0; i < len; i++) {
            procesadorSinaptico = this.procesadorSinaptico[i];

            procesadorSinaptico.calcularSinapsis(this.pesos);
            procesadorSinaptico.calcularError();

            if (procesadorSinaptico.error !== 0) {
                this.hasError = true;
                procesadorSinaptico.reajustarPesos(this.pesos);
                this.funcBack();
            }
        }

        if (this.hasError) {
            this.counterErrors++;

            if (this.counterErrors >= this.LIMIT_ERRORS) {
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

        procesadorSinaptico.calcularSinapsis(this.pesos);

        return procesadorSinaptico.salida();
    }

    getPesos() {
        return this.pesos;
    }

    setPesos(pesosSinapticos) {
        this.pesos = pesosSinapticos;

        return this;
    }

    setFuncionActivacion(funcionActivacion) {
        this.funcionActivacion = funcionActivacion;

        return this;
    }

    private asignarPesos() {
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

        this.setPesos(pesos);
        this.funcBack();
    }
}
