import '@tensorflow/tfjs-node';

import * as tf from '@tensorflow/tfjs';
import { assert } from 'chai';
import { suite, test } from 'mocha-typescript';

// Or if running with GPU:
// import '@tensorflow/tfjs-node-gpu';

@suite
export class TensorflowTest {
    @test
    public async trainXOR() {
        const model: tf.Sequential = tf.sequential();

        const hiddenLayer: tf.layers.Layer = tf.layers.dense({
            units: 2,
            inputShape: [2],
            activation: 'sigmoid'
        });

        const outputLayer: tf.layers.Layer = tf.layers.dense({
            units: 1,
            activation: 'sigmoid'
        });

        model.add(hiddenLayer);
        model.add(outputLayer);

        model.compile({
            loss: 'meanSquaredError',
            optimizer: tf.train.sgd(0.75)
        });

        const inputData = [[0, 0], [1, 0], [0, 1], [1, 1]];
        const input = tf.tensor2d(inputData);
        const output = tf.tensor1d([1, 0, 0, 1]);

        await model.fit(input, output, {
            // batchSize: 8,
            epochs: 1500,
            callbacks: {
                onEpochEnd: async (epoch, log) => {
                    console.log(`Epoch ${epoch}: loss = ${log.loss}`);
                }
            }
        });

        const predictions = (model.predict(input) as tf.Tensor).dataSync();

        for (let index = 0; index < predictions.length; index++) {
            const prediction = predictions[index];

            assert.equal(
                output.get(index),
                Math.round(prediction),
                'input: ' + inputData[index] + ' output: ' + output.get(index)
            );
        }
    }
}
