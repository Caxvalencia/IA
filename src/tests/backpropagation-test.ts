import { assert } from 'chai';
import { suite, test } from 'mocha-typescript';

import { Backpropagation } from '../backpropagation';

@suite
export class PerceptronTest {
    @test
    public testXOR() {
        var data = [
            { input: [0, 0], output: 0 },
            { input: [0, 1], output: 1 },
            { input: [1, 0], output: 1 },
            { input: [1, 1], output: 0 }
        ];

        var XOR = new Backpropagation();
        XOR.addLayer(2) // Entrada o primer capa oculta
            .addLayer(1) // Salida o ultima capa
            .learn(data);

        [
            // Datas
            [[0, 0], 0],
            [[0, 1], 1],
            [[1, 0], 1],
            [[1, 1], 0]
        ].forEach(dataForTest => {
            let output = XOR.process(dataForTest[0])[0];

            assert.equal(
                output,
                dataForTest[1],
                dataForTest[0] + ' -> ' + output
            );
        });
    }
}
