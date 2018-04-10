import { Neuron } from './neuron';

'use strict';
export class Backpropagation {
    LIMIT_ERRORS: number;
    counterErrors: number;
    funcionActivacion: string;
    error: number;
    factorAprendizaje: number;
    capas: any[];

    /**
     * @construtor
     */
    constructor() {
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
    propagarDatos(datos) {
        var salidas = [],
            datosValor = datos.valor;

        this.capas.forEach(function(capa) {
            if (salidas.length > 0) {
                datosValor = salidas;
                salidas = [];
            }

            capa.forEach(function(neurona: Neuron) {
                neurona.addData(datosValor);

                if (!neurona.weights) {
                    neurona.assignWeights();
                }

                neurona.calcularSinapsis();

                if (neurona.neuronasSalida.length > 0) {
                    salidas.push(neurona.salida());
                } else {
                    neurona.calcularError(datos.salida);
                }
            });
        });

        return this;
    }

    /**
     * Metodos publicos
     */
    aprender(datos) {
        var _aprender = function(self) {
            const indiceUltimaCapa = self.capas.length - 1;
            var sumaErrores = 0;

            datos.forEach(function(dato) {
                // Forwardpropagation
                self.propagarDatos(dato);

                // Backpropagation
                self.capas[indiceUltimaCapa].forEach(function(neurona: Neuron) {
                    //Solo con una neurona tenemos acceso a todas las otras
                    neurona.retropropagar();
                    
                    return false;
                });

                //Reajustar pesos
                self.capas.forEach(function(capa) {
                    capa.forEach(function(neurona) {
                        neurona.reajustarPesos();
                        sumaErrores += Math.pow(neurona.error, 2);
                    });
                });

                sumaErrores *= 0.5;
            });

            self.setError(sumaErrores);
        };

        _aprender(this);

        while (this.error > 0.0001) {
            this.counterErrors++;

            if (this.counterErrors >= this.LIMIT_ERRORS) {
                // this.counterErrors = 0;
                return this;
            }

            _aprender(this);
        }

        this.counterErrors = 0;

        return this;
    }

    addCapa(cantNeuronas) {
        let capa: Neuron[] = [];
        let tieneCapas: boolean = this.capas.length > 0;

        for (let i = 0; i < cantNeuronas; i++) {
            capa[i] = new Neuron(tieneCapas);
            capa[i].setLearningFactor(this.factorAprendizaje);
        }

        var indiceNuevaCapa = this.capas.push(capa) - 1;
        var capaAnterior = this.capas[indiceNuevaCapa - 1];

        // Verificar si existe capa anterior
        if (capaAnterior) {
            // Apuntar con cada Neuron de la nueva capa a la anterior
            capa.forEach(function(neurona) {
                neurona.neuronasEntrada = capaAnterior;
            });

            // Apuntar con cada neurona de la capa anterior a la nueva capa
            capaAnterior.forEach(function(neurona) {
                neurona.neuronasSalida = capa;
            });
        }

        return this;
    }

    addNeurona(capa, posicion) {
        var neurona = new Neuron();
        neurona.setLearningFactor(this.factorAprendizaje);

        if (posicion) {
            this.capas[capa][posicion] = neurona;

            return;
        }

        this.capas[capa].push(neurona);
    }

    process(data) {
        let outputs = [];

        this.capas.forEach(function(layer) {
            if (outputs.length > 0) {
                data = outputs;
                outputs = [];
            }

            layer.forEach(function(neuron: Neuron) {
                outputs.push(
                    neuron
                        .addData(data)
                        .calcularSinapsis()
                        .salida()
                );
            });
        });

        return outputs.map(output => {
            return Math.round(output);
        });
    }

    setError(error) {
        this.error = error;

        return this;
    }
}
