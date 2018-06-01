import { assert } from 'chai';
import { suite, test } from 'mocha-typescript';

import { Backpropagation } from '../backpropagation';

@suite
export class BackpropagationTest {
    @test
    public testOR() {
        const data = [
            { input: [0, 0], output: 0 },
            { input: [0, 1], output: 1 },
            { input: [1, 0], output: 1 },
            { input: [1, 1], output: 1 }
        ];

        const OR = new Backpropagation({ epochs: 1000 });
        OR.addLayer(2)
            .addLayer(1)
            .learn(data);

        data.forEach(({ input, output }) => {
            const outputActual = OR.process(input)[0];

            assert.equal(outputActual, output, input + ' -> ' + output);
        });
    }

    //@test
    public testAND() {
        const data = [
            { input: [0, 0], output: 0 },
            { input: [0, 1], output: 0 },
            { input: [1, 0], output: 0 },
            { input: [1, 1], output: 1 }
        ];

        const AND = new Backpropagation({ epochs: 1000 });
        AND.addLayer(2)
            .addLayer(1)
            .learn(data);

        data.forEach(({ input, output }) => {
            const outputActual = AND.process(input)[0];

            assert.equal(outputActual, output, input + ' -> ' + output);
        });
    }

    @test
    public testXOR() {
        const data = [
            { input: [0, 0], output: 0 },
            { input: [0, 1], output: 1 },
            { input: [1, 0], output: 1 },
            { input: [1, 1], output: 0 }
        ];

        const XOR = new Backpropagation({ epochs: 10000 });
        XOR.addLayer(2)
            .addLayer(1)
            .learn(data);

        data.forEach(({ input, output }) => {
            const outputActual = XOR.process(input)[0];

            assert.equal(outputActual, output, input + ' -> ' + output);
        });
    }
}
