var Retropropagacion = ( function() {
	/**
	 * @construtor
	 */
	function Retropropagacion( datos, salidaDeseada ) {
		this.capas = [];
		this.salida = [];

		this.datos = [ this.umbral ].concat( datos );
		this.salidaDeseada = salidaDeseada;
		this.error = 0;
	}

	/**
	 * Metodos privados
	 */
	Retropropagacion.prototype.activacion = function() {
		//Funcion de activacion sigmoide
		return 1 / ( 1 + Math.pow( Math.E, -this.sinapsis ) );
	};

	/**
	 * Metodos publicos
	 */
	Retropropagacion.prototype.reajustarPesos = function( pesos ) {
		var len = pesos.length, i;
		
		for( i = 0; i < len; i++ ) {
			pesos[ i ] = pesos[ i ] + this.factorAprendizaje * this.error * this.datos[ i ];
		}
	};

	Retropropagacion.prototype.addCapa = function( pesos ) {
		var len = pesos.length, i;
		
		for( i = 0; i < len; i++ ) {
			pesos[ i ] = pesos[ i ] + this.factorAprendizaje * this.error * this.datos[ i ];
		}
	};

	Retropropagacion.prototype.calcularSinapsis = function( pesos ) {
		var len = pesos.length, i;
		this.sinapsis = 0;

		for( i = 0; i < len; i++ ) {
			this.sinapsis += this.datos[ i ] * pesos[ i ];
		}

		return this;
	};

	Retropropagacion.prototype.calcularError = function( funcActivacion ) {
		this.error = this.salidaDeseada - this.salida();
		return this;
	};

	return Retropropagacion;
})();