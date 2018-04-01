export class Perceptron {
    hasError: boolean;
    info: any[];
    pesos: any;
    rangoPesos: { MIN: number; MAX: number };
    factorAprendizaje: number;

    constructor() {
        this.factorAprendizaje = 0.5;
        this.rangoPesos = { MIN: -5, MAX: 4.9 };
        this.pesos = null;
        this.info = [];
        this.hasError = false;
    }

    /**
     * Metodos privados
     */
    salidaObtenida(sinapsis) {
        return sinapsis >= 0 ? 1 : 0;
    }

    asignarPesos() {
        this.pesos = new Array(this.info[0].datos.length);

        var len = this.pesos.length,
            i,
            rango = this.rangoPesos.MAX - this.rangoPesos.MIN;

        for (i = 0; i < len; i++) {
            while (!this.pesos[i])
                this.pesos[i] = parseFloat(
                    (Math.random() * rango + this.rangoPesos.MIN).toFixed(4)
                );
        }
    }

    reajustarPesos(info) {
        var len = this.pesos.length,
            i;

        for (i = 0; i < len; i++) {
            this.pesos[i] =
                this.pesos[i] +
                this.factorAprendizaje * info.error * info.datos[i];
        }
    }

    calcularSinapsis(info) {
        var len = this.pesos.length,
            i;
        info.sinapsis = 0;

        for (i = 0; i < len; i++) {
            info.sinapsis += info.datos[i] * this.pesos[i];
        }

        return this;
    }

    calcularError(info) {
        info.error = info.salida - this.salidaObtenida(info.sinapsis);
        return this;
    }

    /**
     * Metodos publicos
     */
    addDatos(datos, salida) {
        datos = [1].concat(datos);

        this.info.push({
            datos: datos,
            salida: salida,
            sinapsis: 0,
            error: 0
        });

        return this;
    }

    aprender() {
        if (this.info.length === 0) return;

        if (!this.pesos) this.asignarPesos();

        var info = null,
            i,
            len = this.info.length;

        this.hasError = false;

        for (i = 0; i < len; i++) {
            info = this.info[i];
            this.calcularSinapsis(info);
            this.calcularError(info);

            if (info.error !== 0) {
                this.hasError = true;
                this.reajustarPesos(info);
            }
        }

        if (this.hasError) return this.aprender();
    }

    procesar(datos) {
        datos = [1].concat(datos);

        var info = {
            datos: datos,
            salida: null,
            sinapsis: 0,
            error: 0
        };

        this.calcularSinapsis(info);
        this.calcularError(info);

        return this.salidaObtenida(info.sinapsis);
    }

    /**
     * Getters y setters
     */
    getPesos() {
        return this.pesos;
    }
}
