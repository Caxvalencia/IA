var Perceptron = ( function( ProcesadorSinaptico ) {
	/**
	 * @construtor
	 */
	function Perceptron() {
		this.rangoPesos = { MIN: -5, MAX: 4.9 };
		this.hasError = false;
		this.procesadorSinaptico = [];

		this.counterLimitCalls = 0;
		this.LIMIT_CALLS = 10000;

		this._pesos = null;
		this._funcBack = function() {};
	}

	/**
	 * Metodos privados
	 */
	function asignarPesos() {
		var len = this.procesadorSinaptico[ 0 ].datos.length,
			pesos = new Array( len ),
			rango = this.rangoPesos.MAX - this.rangoPesos.MIN,
			i;

		for( i = 0; i < len; i++ ) {
			while( !pesos[ i ] )
				pesos[ i ] = parseFloat( ( Math.random() * rango + this.rangoPesos.MIN ).toFixed( 4 ) );
		}

		this.pesos( pesos );
		this._funcBack( this );
	};

	/**
	 * Metodos publicos
	 */
	Perceptron.prototype.addDatos = function( datos, salida ) {
		if( datos[ 0 ] !== undefined )
			if( datos[ 0 ][ 0 ] !== undefined ) {
				for( var i = 0; i <= datos.length; i++ ) {
					this.procesadorSinaptico.push( new ProcesadorSinaptico( datos[ i ], salida[ i ] ) );
				}
			} else
				this.procesadorSinaptico.push( new ProcesadorSinaptico( datos, salida ) );
		return this;
	};

	Perceptron.prototype.aprender = function() {
		if( this.procesadorSinaptico.length === 0 )
			return;

		if( !this._pesos )
			asignarPesos.call( this );

		var procesadorSinaptico = null,
			len = this.procesadorSinaptico.length,
			i;
		
		this.hasError = false;

		for( i = 0; i < len; i++ ) {
			procesadorSinaptico = this.procesadorSinaptico[ i ];

			procesadorSinaptico.calcularSinapsis( this._pesos );
			procesadorSinaptico.calcularError();

			if( procesadorSinaptico.error !== 0 ) {
				this.hasError = true;
				procesadorSinaptico.reajustarPesos( this._pesos );
				this._funcBack( this );
			}
		}

		if( this.hasError ) {
			this.counterLimitCalls++;

			if( this.counterLimitCalls >= this.LIMIT_CALLS ) {
				this.counterLimitCalls = 0;
				return this;
			}

			return this.aprender();
		}

		return this;
	};

	Perceptron.prototype.procesar = function( datos ) {
		var procesadorSinaptico = new ProcesadorSinaptico( datos, null );
		procesadorSinaptico.calcularSinapsis( this._pesos );

		return procesadorSinaptico.salida();
	};

	Perceptron.prototype.funcBack = function( funcBack ) {
		this._funcBack = funcBack || function() {};
		return this;
	};

	/**
	 * Getters y setters
	 */
	Perceptron.prototype.pesos = function( pesosSinapticos ) {
		if( pesosSinapticos ) {
			this._pesos = pesosSinapticos;
			return this;
		}

		return this._pesos;
	};

	return Perceptron;
})( ProcesadorSinaptico );