var ProcesadorSinaptico = ( function() {
	/**
	 * @construtor
	 */
	function ProcesadorSinaptico( datos, salidaDeseada, funcionActivacion ) {
		this.factorAprendizaje = 0.5;
		this.umbral = 1;

		this.datos = datos ? [ this.umbral ].concat( datos ) : [ this.umbral ];
		this.salidaDeseada = salidaDeseada;
		this.sinapsis = 0;
		this.error = 0;
		this.funcionActivacion = funcionActivacion;
	}

	/**
	 * Metodos privados
	 */

	/**
	 * Metodos publicos
	 */
	ProcesadorSinaptico.prototype.salida = function() {
		if( !this.funcionActivacion ) return this.sinapsis >= 0 ? 1 : 0;
		
		//Funcion de activacion sigmoidal binaria
		if( this.funcionActivacion === 'sigmoidal' ) return 1 / ( 1 + Math.pow( Math.E, -this.sinapsis ) );
	};

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

	ProcesadorSinaptico.prototype.calcularError = function() {
		this.error = this.salidaDeseada - this.salida();
		return this;
	};

	/**
	 * Getters y setters
	 */
	ProcesadorSinaptico.prototype.setDatos = function( datos ) {
		this.datos = datos;
		return this;
	};

	ProcesadorSinaptico.prototype.setSalidaDeseada = function( salidaDeseada ) {
		this.salidaDeseada = salidaDeseada;
		return this;
	};

	return ProcesadorSinaptico;
})();