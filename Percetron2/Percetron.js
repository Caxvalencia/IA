var Percetron = ( function( ProcesadorSinaptico ) {
	/**
	 * @construtor
	 */
	function Percetron() {
		this.rangoPesos = { MIN: -5, MAX: 4.9 };
		this.hasError = false;
		this.procesadorSinaptico = [];

		this._pesos = null;
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
	};

	/**
	 * Metodos publicos
	 */
	Percetron.prototype.addDatos = function( datos, salida ) {
		this.procesadorSinaptico.push( new ProcesadorSinaptico( datos, salida ) );
		return this;
	};

	Percetron.prototype.aprender = function() {
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
			}
		}

		if( this.hasError )
			return this.aprender();
	};

	Percetron.prototype.procesar = function( datos ) {
		var procesadorSinaptico = new ProcesadorSinaptico( datos, null );
		procesadorSinaptico.calcularSinapsis( this._pesos );

		return procesadorSinaptico.salida();
	};

	/**
	 * Getters y setters
	 */
	Percetron.prototype.pesos = function( pesosSinapticos ) {
		if( pesosSinapticos ) {
			this._pesos = pesosSinapticos;
			return this;
		}

		return this._pesos;
	};

	return Percetron;
})( ProcesadorSinaptico );