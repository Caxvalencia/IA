import { sigmoidal } from './sigmoidal.function';
import { reLU } from './relu.function';
import { hyperbolicTangent, prime } from './hyperbolic-tangent.function';
import { binary } from './binary.function';

export enum ActivationFunctionType {
    BINARY = 'BINARY',
    RELU = 'RELU',
    SIGMOIDAL = 'SIGMOIDAL',
    HYPERBOLIC_TANGENT = 'HYPERBOLIC_TANGENT'
}

export class ActivationFunction {
    protected default: string;
    private callback: any;

    constructor(
        functionName: ActivationFunctionType = ActivationFunctionType.BINARY
    ) {
        this.default = functionName;
        this.callback = this.getCallback();
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
    process(synapse: number) {
        return this.callback(synapse);
    }

    private getCallback(): any {
        const callback = {
            [ActivationFunctionType.BINARY]: binary,
            [ActivationFunctionType.RELU]: reLU,
            [ActivationFunctionType.SIGMOIDAL]: sigmoidal,
            [ActivationFunctionType.HYPERBOLIC_TANGENT]: hyperbolicTangent
        };

        this.callback = callback[this.default];
    }
}
