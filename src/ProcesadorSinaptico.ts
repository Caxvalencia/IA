export class ProcesadorSinaptico {
    funcionActivacion: any;
    error: number;
    sinapsis: number;
    salidaDeseada: any;
    datos: number[];
    umbral: number;
    factorAprendizaje: number;

    constructor(datos, salidaDeseada, funcionActivacion) {
        this.factorAprendizaje = 0.5;
        this.umbral = 1;

        this.datos = datos ? [this.umbral].concat(datos) : [this.umbral];
        this.salidaDeseada = salidaDeseada;
        this.sinapsis = 0;
        this.error = 0;
        this.funcionActivacion = funcionActivacion;
    }

    salida() {
        if (!this.funcionActivacion) return this.sinapsis >= 0 ? 1 : 0;

        //Funcion de activacion sigmoidal binaria
        if (this.funcionActivacion === 'sigmoidal')
            return 1 / (1 + Math.pow(Math.E, -this.sinapsis));
    }

    reajustarPesos(pesos) {
        var len = pesos.length,
            i;

        for (i = 0; i < len; i++) {
            pesos[i] =
                pesos[i] + this.factorAprendizaje * this.error * this.datos[i];
        }
    }

    calcularSinapsis(pesos) {
        var len = pesos.length,
            i;
        this.sinapsis = 0;

        for (i = 0; i < len; i++) {
            this.sinapsis += this.datos[i] * pesos[i];
        }

        return this;
    }

    calcularError() {
        this.error = this.salidaDeseada - this.salida();
        return this;
    }

    /**
     * Getters y setters
     */
    setDatos(datos) {
        this.datos = datos;
        return this;
    }

    setSalidaDeseada(salidaDeseada) {
        this.salidaDeseada = salidaDeseada;
        return this;
    }
}
