import { assert } from 'chai';
import { suite, test } from 'mocha-typescript';

import { Backpropagation } from '../backpropagation';

@suite
export class BackpropagationTest {
    @test
    public testXOR() {
        let data = [
            { input: [0, 0], output: 0 },
            { input: [0, 1], output: 1 },
            { input: [1, 0], output: 1 },
            { input: [1, 1], output: 0 }
        ];

        let XOR = new Backpropagation();
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
            const data = dataForTest[0];
            const outputExpected = dataForTest[1];
            const output = XOR.process(data)[0];

            assert.equal(output, outputExpected, data + ' -> ' + output);
        });
    }
}
