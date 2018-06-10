import { assert } from 'chai';
import { suite, test } from 'mocha-typescript';

import { Backpropagation } from '../backpropagation';

@suite
export class BackpropagationImportExportTest {
    @test
    public importExportModel() {
        const model = {
            layers: [3, 1],
            thresholds: [[-0.3658, -0.0281, 0.2527], [-0.2412]],
            weights: [
                [[7.76849, -4.66939], [-4.12036, 5.06214], [6.66429, 7.7002]],
                [[-11.90173, -12.22545, 15.43677]]
            ]
        };

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
}
