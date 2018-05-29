export namespace Sigmoidal {
    export function activation(synapse) {
        return 1 / (1 + Math.pow(Math.E, -synapse));
    }

    export function prime(synapse) {
        const output = activation(synapse);

        return output - output * output;
    }
}
