import { assert } from 'chai';
import { suite, test } from 'mocha-typescript';

import { Backpropagation } from '../backpropagation';
import { ActivationFunctionType } from '../activation-functions/activation-function';

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

    @test
    public testAND() {
        const data = [
            { input: [0, 0], output: 0 },
            { input: [0, 1], output: 0 },
            { input: [1, 0], output: 0 },
            { input: [1, 1], output: 1 }
        ];

        const AND = new Backpropagation({ epochs: 1500 });
        AND.addLayer(3)
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

        const XOR = new Backpropagation({ epochs: 5000 });
        XOR.addLayer(3)
            .addLayer(1)
            .learn(data);

        data.forEach(({ input, output }) => {
            const outputActual = XOR.process(input)[0];

            assert.equal(outputActual, output, input + ' -> ' + output);
        });
    }

    @test
    public testHyperbolicTangentForXOR() {
        const data = [
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
            .learn(data);

        data.forEach(({ input, output }) => {
            const outputActual = XOR.process(input)[0];

            assert.equal(outputActual, output, input + ' -> ' + output);
        });
    }

    @test
    public testImportConfiguration() {
        const data = [
            { input: [0, 0], output: 0 },
            { input: [0, 1], output: 1 },
            { input: [1, 0], output: 1 },
            { input: [1, 1], output: 0 }
        ];

        const model = {
            layers: [3, 1],
            weights: [
                [
                    [5.747799873352051, -3.111999988555908],
                    [-2.653899908065796, 3.5102999210357666],
                    [5.576099872589111, 5.699999809265137]
                ],
                [[-9.125900268554688, -9.922599792480469, 12.401000022888184]]
            ]
        };

        const XOR = new Backpropagation({ epochs: 5000 });
        XOR.importModel(model).learn(data);

        data.forEach(({ input, output }) => {
            const outputActual = XOR.process(input)[0];

            assert.equal(outputActual, output, input + ' -> ' + output);
        });
    }
}
