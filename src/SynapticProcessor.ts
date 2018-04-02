export class SynapticProcessor {
    activationFunction: string;
    error: number;
    synapse: number;
    expectedOutput: any;
    data: number[];
    threshold: number;
    learningFactor: number;

    constructor(data, expectedOutput, activationFunction) {
        this.learningFactor = 0.5;
        this.threshold = 1;

        this.expectedOutput = expectedOutput;
        this.synapse = 0;
        this.error = 0;
        this.activationFunction = activationFunction;

        this.setDatos(data);
    }

    salida() {
        if (!this.activationFunction) {
            return this.synapse >= 0 ? 1 : 0;
        }

        //Funcion de activacion sigmoidal binaria
        if (this.activationFunction === 'sigmoidal') {
            return 1 / (1 + Math.pow(Math.E, -this.synapse));
        }
    }

    reajustarPesos(pesos) {
        for (let i = 0; i < pesos.length; i++) {
            pesos[i] += this.learningFactor * this.error * this.data[i];
        }
    }

    calcularSinapsis(pesos) {
        this.synapse = 0;

        for (let i = 0; i < pesos.length; i++) {
            this.synapse += this.data[i] * pesos[i];
        }

        return this;
    }

    calcularError() {
        this.error = this.expectedOutput - this.salida();

        return this;
    }

    setDatos(datos) {
        this.data = datos ? [this.threshold].concat(datos) : [this.threshold];

        return this;
    }

    setSalidaDeseada(salidaDeseada) {
        this.expectedOutput = salidaDeseada;

        return this;
    }
}
