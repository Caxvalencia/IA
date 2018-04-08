var Backpropagation = ( function( Neurona ) {
	"use strict";

	/**
	 * @construtor
	 */
	function Backpropagation() {
		this.capas = [];
		this.factorAprendizaje = 0.25;
		this.error = 0;
		this.funcionActivacion = 'sigmoidal';

		this.counterErrors = 0;
		this.LIMIT_ERRORS = 100000;
	}

	/**
	 * Metodos privados
	 */
	Backpropagation.prototype.propagarDatos = function( datos ) {
		var salidas = [],
			datosValor = datos.valor;

		this.capas.forEach( function( capa ) {
			if( salidas.length > 0 ) {
				datosValor = salidas;
				salidas = [];
			}

			capa.forEach( function( neurona ) {
				neurona.setDatos( datosValor );

				if( !neurona._pesos )
					neurona.crearPesos();

				neurona.calcularSinapsis();

				if( neurona.neuronasSalida.length > 0 ) {
					salidas.push( neurona.salida() );
				} else {
					neurona.calcularError( datos.salida );
				}
			});
		});

		return this;
	};

	/**
	 * Metodos publicos
	 */
	Backpropagation.prototype.aprender = function( datos ) {
		var _aprender = function( self ) {
			const indiceUltimaCapa = self.capas.length - 1;
			var sumaErrores = 0;

			datos.forEach( function( dato ) {
				// Propagacion
				self.propagarDatos( dato );

				// Retropropagacion
				self.capas[ indiceUltimaCapa ].forEach( function( neurona ) {
					//Solo con una neurona tenemos acceso a todas las otras
					neurona.retropropagar();
					return false;
				});

				//Reajustar pesos
				self.capas.forEach( function( capa ) {
					capa.forEach( function( neurona ) {
						neurona.reajustarPesos();
						sumaErrores += Math.pow( neurona.error, 2 );
					});
				});

				sumaErrores *= 0.5;
			});

			self.setError( sumaErrores );
		}

		_aprender( this );

		while( this.error > 0.0001 ) {
			this.counterErrors++;

			if( this.counterErrors >= this.LIMIT_ERRORS ) {
				// this.counterErrors = 0;
				return this;
			}

			_aprender( this );
		}

		this.counterErrors = 0;
		return this;
	};

	Backpropagation.prototype.addCapa = function( cantNeuronas ) {
		var self = this,
			i,
			capa = [],
			tieneCapas = self.capas.length > 0;
		
		for( i = 0; i < cantNeuronas; i++ ) {
			capa[ i ] = new Neurona( tieneCapas );
			capa[ i ].setFactorAprendizaje( self.factorAprendizaje );
		}

		var indiceNuevaCapa = self.capas.push( capa ) - 1;
		var capaAnterior = self.capas[ indiceNuevaCapa - 1 ];
		
		// Verificar si existe capa anterior
		if( capaAnterior ) {
			// Apuntar con cada neurona de la nueva capa a la anterior
			capa.forEach( function( neurona ) {
				neurona.neuronasEntrada = capaAnterior;
			});
			
			// Apuntar con cada neurona de la capa anterior a la nueva capa
			capaAnterior.forEach( function( neurona ) {
				neurona.neuronasSalida = capa;
			});
		}

		return self;
	};

	Backpropagation.prototype.addNeurona = function( capa, posicion ) {
		var neurona = new Neurona();
		neurona.setFactorAprendizaje( self.factorAprendizaje );

		if( posicion ) this.capas[ capa ][ posicion ] = neurona;
		else this.capas[ capa ].push( neurona );
	};

	Backpropagation.prototype.procesar = function( datos ) {
		var salidas = [];

		this.capas.forEach( function( capa ) {
			if( salidas.length > 0 ) {
				datos = salidas;
				salidas = [];
			}

			capa.forEach( function( neurona ) {
				salidas.push( neurona.setDatos( datos ).calcularSinapsis().salida() );
			});
		});

		return salidas.map( function( salida ) {
			return Math.round( salida );
			
			if( salida >= 0.5 ) return 1;
			else return 0;
		});
	};

	Backpropagation.prototype.setError = function( error ) {
		this.error = error;
		return this;
	};

	return Backpropagation;
})( Neuron );