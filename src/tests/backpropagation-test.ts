import { assert } from 'chai';
import { suite, test } from 'mocha-typescript';

import { ActivationFunctionType } from '../activation-functions/activation-function';
import { Backpropagation } from '../backpropagation';

@suite
export class BackpropagationTest {
    @test
    public testOR() {
        const dataset = [
            { input: [0, 0], output: 0 },
            { input: [0, 1], output: 1 },
            { input: [1, 0], output: 1 },
            { input: [1, 1], output: 1 }
        ];

        const OR = new Backpropagation({ epochs: 1000 });
        OR.addLayer(2)
            .addLayer(1)
            .learn(dataset);

        dataset.forEach(({ input, output }) => {
            const outputActual = OR.process(input)[0];

            assert.equal(outputActual, output, input + ' -> ' + output);
        });
    }

    @test
    public testAND() {
        const dataset = [
            { input: [0, 0], output: 0 },
            { input: [0, 1], output: 0 },
            { input: [1, 0], output: 0 },
            { input: [1, 1], output: 1 }
        ];

        const AND = new Backpropagation({ epochs: 1500 });
        AND.addLayer(3)
            .addLayer(1)
            .learn(dataset);

        dataset.forEach(({ input, output }) => {
            const outputActual = AND.process(input)[0];

            assert.equal(outputActual, output, input + ' -> ' + output);
        });
    }

    @test
    public testXOR() {
        const dataset = [
            { input: [0, 0], output: 1 },
            { input: [0, 1], output: 0 },
            { input: [1, 0], output: 0 },
            { input: [1, 1], output: 1 }
        ];

        const XOR = new Backpropagation({ epochs: 5000 });
        XOR.addLayer(3)
            .addLayer(1)
            .learn(dataset);

        dataset.forEach(({ input, output }) => {
            const outputActual = XOR.process(input)[0];

            assert.equal(outputActual, output, input + ' -> ' + output);
        });
    }

    @test
    public testHyperbolicTangentForXOR() {
        const dataset = [
            { input: [-1, -1], output: -1 },
            { input: [-1, 1], output: 1 },
            { input: [1, -1], output: 1 },
            { input: [1, 1], output: -1 }
        ];

        const XOR = new Backpropagation({
            epochs: 5000,
            activationFunction: ActivationFunctionType.HYPERBOLIC_TANGENT
        });
        XOR.addLayer(3)
            .addLayer(1)
            .learn(dataset);

        dataset.forEach(({ input, output }) => {
            const outputActual = XOR.process(input)[0];

            assert.equal(outputActual, output, input + ' -> ' + output);
        });
    }
}
