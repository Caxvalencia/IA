/// <reference path="../../node_modules/mocha-typescript/globals.d.ts" />

import { assert } from 'chai';
import { suite, test } from 'mocha-typescript';

import { Perceptron } from '../Perceptron';

@suite
export class PerceptronTest {
    @test
    public testAND() {
        let perceptron = new Perceptron();
        let data = [[[0, 0], 0], [[0, 1], 0], [[1, 0], 0], [[1, 1], 1]];

        data.forEach(data => {
            perceptron.addDatos(data[0], data[1]).aprender();
        });

        data.forEach(data => {
            assert.equal(
                data[1],
                perceptron.procesar(data[0]),
                data[0] + ' -> ' + data[1]
            );
        });
    }

    @test
    public testOR() {
        let perceptron = new Perceptron();
        let data = [[[0, 0], 0], [[0, 1], 1], [[1, 0], 1], [[1, 1], 1]];

        data.forEach(data => {
            perceptron.addDatos(data[0], data[1]).aprender();
        });

        data.forEach(data => {
            assert.equal(
                data[1],
                perceptron.procesar(data[0]),
                data[0] + ' -> ' + data[1]
            );
        });
    }

    @test
    public testFailXOR() {
        let perceptron = new Perceptron();
        let data = [[[0, 0], 1], [[0, 1], 0], [[1, 0], 0], [[1, 1], 1]];

        assert.throws(() => {
            data.forEach(data => {
                perceptron.addDatos(data[0], data[1]).aprender();
            });
        });
    }
}
