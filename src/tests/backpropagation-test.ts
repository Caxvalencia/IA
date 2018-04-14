import { assert } from 'chai';
import { suite, test } from 'mocha-typescript';

import { Backpropagation } from '../backpropagation';

@suite
export class PerceptronTest {
    @test
    public testXOR() {
        var XOR = [
            { input: [0, 0], output: 0 },
            { input: [0, 1], output: 1 },
            { input: [1, 0], output: 1 },
            { input: [1, 1], output: 0 }
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
    }
}
