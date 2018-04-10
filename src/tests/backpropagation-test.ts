import { assert } from 'chai';
import { suite, test } from 'mocha-typescript';

import { Backpropagation } from '../backpropagation';

@suite
export class PerceptronTest {
    @test
    public testXOR() {
        /**
         * Funcion XOR
         */
        var XOR = [
            { valor: [0, 0], salida: 0 },
            { valor: [0, 1], salida: 1 },
            { valor: [1, 0], salida: 1 },
            { valor: [1, 1], salida: 0 }
        ];

        var func_XOR = new Backpropagation();

        func_XOR
            .addLayer(2) // Entrada o primer capa oculta
            .addLayer(1) // Salida o ultima capa
            .learn(XOR);

        var results = [
            func_XOR.process([0, 0])[0],
            func_XOR.process([0, 1])[0],
            func_XOR.process([1, 0])[0],
            func_XOR.process([1, 1])[0]
        ];

        assert.isTrue(results[0] === 0, [0, 0] + ' -> ' + results[0]);

        // console.log('XOR' + '\n',
        //     , , '\n',
        //     [0, 1], '->', results[1], results[1] === 1, '\n',
        //     [1, 0], '->', results[2], results[2] === 1, '\n',
        //     [1, 1], '->', results[3], results[3] === 0, '\n',
        //     func_XOR
        // );

        // func_XOR.capas.forEach(function (capa, idxCapa) {
        //     capa.forEach(function (neurona, idx) {
        //         console.log(idxCapa, idx, neurona);
        //     });
        // });
    }
}
