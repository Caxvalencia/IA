import { assert } from 'chai';
import { suite, test } from 'mocha-typescript';

import { Backpropagation } from '../backpropagation';

@suite
export class BackpropagationImportExportTest {
    @test
    public importExportModel() {
        const model = this.modelData();
        const XOR = new Backpropagation();
        const modelExported = XOR.importModel(model).exportModel();

        assert.sameDeepMembers(model.layers, modelExported.layers, 'Layers');
        assert.sameDeepMembers(model.weights, modelExported.weights, 'Weights');
        assert.sameDeepMembers(
            model.thresholds,
            modelExported.thresholds,
            'Thresholds'
        );
    }

    @test('given one model data imported should relearning')
    public givenOneModelDataImportedShouldRelearning() {
        const model = this.modelData();

        const dataset = [
            { input: [0, 0], output: 1 },
            { input: [0, 1], output: 0 },
            { input: [1, 0], output: 0 },
            { input: [1, 1], output: 1 }
        ];

        const resultExpected = [
            { input: [0, 0], output: 0.9822183389766372 },
            { input: [0, 1], output: 0.015593981668139112 },
            { input: [1, 0], output: 0.015273412902442525 },
            { input: [1, 1], output: 0.9951732837775824 }
        ];

        const XOR = new Backpropagation();
        XOR.importModel(model).learn(dataset);

        assert.equal(XOR.error, 2.68879322274913e-10, 'error');

        resultExpected.forEach(({ input, output }) => {
            const outputActual = XOR.process(input)[0];

            assert.equal(outputActual, output, input + ' -> ' + output);
        });
    }

    private modelData() {
        return {
            layers: [3, 1],
            thresholds: [[-0.3658, -0.0281, 0.2527], [-0.2412]],
            weights: [
                [[7.76849, -4.66939], [-4.12036, 5.06214], [6.66429, 7.7002]],
                [[-11.90173, -12.22545, 15.43677]]
            ]
        };
    }
}
