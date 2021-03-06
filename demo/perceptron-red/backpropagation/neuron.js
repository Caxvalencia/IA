var Neuron = (function() {
    /**
     * @construtor
     */
    function Neuron(_isOculta, datos) {
        this.factorAprendizaje = 0.5;
        this._isOculta = _isOculta;

        this._pesos = null;
        this.sinapsis = 0;
        this.error = 0;

        this.neuronasSalida = [];
        this.neuronasEntrada = [];

        this.umbral = 1;
        this._datos = [];
        this.setDatos(datos);
    }

    /**
     * Metodos privados
     */
    Neuron.prototype.crearPeso = function() {
        var rangoPesos = { MIN: -5, MAX: 4.9 },
            peso = 0,
            rango = rangoPesos.MAX - rangoPesos.MIN;

        while (!peso)
            peso = parseFloat(
                (Math.random() * rango + rangoPesos.MIN).toFixed(4)
            );

        return peso;
    };

    Neuron.prototype.crearPesos = function() {
        var i,
            len = this._datos.length,
            pesos = new Array(len);

        for (i = 0; i < len; i++) {
            while (!pesos[i]) pesos[i] = this.crearPeso();
        }

        this._pesos = pesos;
        return this;
    };

    Neuron.prototype.calcularSinapsis = function() {
        var len = this._pesos.length,
            i;
        this.sinapsis = 0;

        for (i = 0; i < len; i++) {
            this.sinapsis += this._datos[i] * this._pesos[i];
        }

        return this;
    };

    /**
     * Metodos publicos
     */
    Neuron.prototype.retropropagar = function() {
        // Error en las capas ocultas
        this.neuronasEntrada.forEach(function(neurona, idx) {
            neurona.calcularErrorOculto(idx);
        });

        if (this.neuronasEntrada.length > 0) {
            this.neuronasEntrada[0].retropropagar();
        }
    };

    Neuron.prototype.reajustarPesos = function() {
        var len = this._pesos.length,
            i;

        for (i = 0; i < len; i++) {
            this._pesos[i] =
                this._pesos[i] +
                this.factorAprendizaje * this.error * this._datos[i];
        }
    };

    Neuron.prototype.salida = function() {
        //Funcion de activacion sigmoidal binaria
        return 1 / (1 + Math.pow(Math.E, -this.sinapsis));
    };

    Neuron.prototype.calcularError = function(salidaDeseada) {
        var salidaObtenida = this.salida();
        this.error =
            (salidaDeseada - salidaObtenida) *
            (1 - salidaObtenida) *
            salidaObtenida;
        return this;
    };

    Neuron.prototype.calcularErrorOculto = function(idx) {
        var salidaObtenida = this.salida(),
            sumatoriaError = 0;

        this.neuronasSalida.forEach(function(neurona) {
            sumatoriaError += neurona.error * neurona._pesos[idx];
        });

        this.error = sumatoriaError * (1 - salidaObtenida) * salidaObtenida;
        return this;
    };

    /**
     * Getters y setters
     */
    Neuron.prototype.setFactorAprendizaje = function(factor) {
        this.factorAprendizaje = factor;
        return this;
    };

    Neuron.prototype.setDatos = function(datos) {
        if (datos) {
            this._datos = [].concat(datos);
            this._datos.push(this.umbral);
        }

        return this;
    };

    return Neuron;
})();
