import { sigmoidal } from './sigmoidal.function';
import { reLU } from './relu.function';
import { hyperbolicTangent } from './hyperbolic-tangent.function';
import { binary } from './binary.function';

export enum ActivationFunctionType {
    BINARY = 'BINARY',
    RELU = 'RELU',
    SIGMOIDAL = 'SIGMOIDAL',
    HYPERBOLIC_TANGENT = 'HYPERBOLIC_TANGENT'
}

export class ActivationFunction {
    protected default: string;

    constructor(
        functionName: ActivationFunctionType = ActivationFunctionType.BINARY
    ) {
        this.default = functionName;
    }

    /**
     * @static
     * @param {string} functionName 
     * @param {ActivationFunctionType} [defaultFunction] 
     * @returns  
     */
    static process(
        functionName: string,
        defaultFunction?: ActivationFunctionType
    ) {
        return new ActivationFunction(defaultFunction).process(functionName);
    }

    /**
     * @param {string} functionName 
     * @returns  
     */
    process(functionName: string) {
        const callback = {
            [ActivationFunctionType.BINARY]: binary,
            [ActivationFunctionType.RELU]: reLU,
            [ActivationFunctionType.SIGMOIDAL]: sigmoidal,
            [ActivationFunctionType.HYPERBOLIC_TANGENT]: hyperbolicTangent
        };

        return callback[functionName] || callback[this.default];
    }
}
