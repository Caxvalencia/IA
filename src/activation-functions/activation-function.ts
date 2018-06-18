import { Binary } from './binary.function';
import { HyperbolicTangent } from './hyperbolic-tangent.function';
import { ReLU } from './relu.function';
import { Sigmoidal } from './sigmoidal.function';

export enum ActivationFunctionType {
    BINARY = 'BINARY',
    RELU = 'RELU',
    SIGMOIDAL = 'SIGMOIDAL',
    HYPERBOLIC_TANGENT = 'HYPERBOLIC_TANGENT'
}

const callback = {
    [ActivationFunctionType.BINARY]: Binary,
    [ActivationFunctionType.RELU]: ReLU,
    [ActivationFunctionType.SIGMOIDAL]: Sigmoidal,
    [ActivationFunctionType.HYPERBOLIC_TANGENT]: HyperbolicTangent
};

export class ActivationFunction {
    protected default: string;
    private callback: Function;
    private callbackPrime: Function;

    private cache;
    private cachePrime;

    constructor(
        functionName: ActivationFunctionType = ActivationFunctionType.BINARY
    ) {
        this.default = functionName;
        this.setCallback();
        this.setCallbackPrime();

        this.cache = {};
        this.cachePrime = {};
    }

    /**
     * @static
     * @param {string} functionName
     * @returns
     */
    static init(functionName: ActivationFunctionType) {
        return new ActivationFunction(functionName);
    }

    /**
     * @param {number} synapse
     * @returns
     */
    activation(synapse: number) {
        if (!this.cache[synapse]) {
            this.cache[synapse] = this.callback(synapse);
        }

        return this.cache[synapse];
    }

    /**
     * @param {number} synapse
     * @returns
     */
    prime(synapse: number) {
        if (!this.cachePrime[synapse]) {
            this.cachePrime[synapse] = this.callbackPrime(synapse);
        }

        return this.cachePrime[synapse];
    }

    private setCallback() {
        this.callback = callback[this.default].activation;
    }

    private setCallbackPrime() {
        this.callbackPrime = callback[this.default].prime;
    }
}
