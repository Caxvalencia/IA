var ProcesadorSinaptico = ( function() {
	/**
	 * @construtor
	 */
	function ProcesadorSinaptico( datos, salidaDeseada ) {
		this.factorAprendizaje = 0.5;
		this.umbral = 1;

		this.datos = [ this.umbral ].concat( datos );
		this.salidaDeseada = salidaDeseada;
		this.sinapsis = 0;
		this.error = 0;
	}

	/**
	 * Metodos privados
	 */
	ProcesadorSinaptico.prototype.salida = function() {
		return this.sinapsis >= 0 ? 1 : 0;
	};

	/**
	 * Metodos publicos
	 */
	ProcesadorSinaptico.prototype.reajustarPesos = function( pesos ) {
		var len = pesos.length, i;
		
		for( i = 0; i < len; i++ ) {
			pesos[ i ] = pesos[ i ] + this.factorAprendizaje * this.error * this.datos[ i ];
		}
	};

	ProcesadorSinaptico.prototype.calcularSinapsis = function( pesos ) {
		var len = pesos.length, i;
		this.sinapsis = 0;

		for( i = 0; i < len; i++ ) {
			this.sinapsis += this.datos[ i ] * pesos[ i ];
		}

		return this;
	};

	ProcesadorSinaptico.prototype.calcularError = function( funcActivacion ) {
		this.error = this.salidaDeseada - this.salida();
		return this;
	};

	ProcesadorSinaptico.prototype.addProcesadorSinaptico = function( funcActivacion ) {
		return this;
	};

	return ProcesadorSinaptico;
})();