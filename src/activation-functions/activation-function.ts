import { Binary } from './binary.function';
import { HyperbolicTangent } from './hyperbolic-tangent.function';
import { ReLU } from './relu.function';
import { Sigmoidal } from './sigmoidal.function';

declare type CacheType = {
    activation: { sypnase: number; value: number };
    prime: { sypnase: number; value: number };
};

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
    private cache: CacheType;

    constructor(
        functionName: ActivationFunctionType = ActivationFunctionType.BINARY
    ) {
        this.default = functionName;
        this.setCallback();
        this.setCallbackPrime();

        this.cache = {
            activation: { sypnase: null, value: null },
            prime: { sypnase: null, value: null }
        };
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
        if (this.cache.activation.sypnase !== synapse) {
            this.cache.activation.sypnase = synapse;
            this.cache.activation.value = this.callback(synapse);
        }

        return this.cache.activation.value;
    }

    /**
     * @param {number} synapse
     * @returns
     */
    prime(synapse: number) {
        if (this.cache.prime.sypnase !== synapse) {
            this.cache.prime.sypnase = synapse;
            this.cache.prime.value = this.callbackPrime(synapse);
        }

        return this.cache.prime.value;
    }

    private setCallback() {
        this.callback = callback[this.default].activation;
    }

    private setCallbackPrime() {
        this.callbackPrime = callback[this.default].prime;
    }
}
