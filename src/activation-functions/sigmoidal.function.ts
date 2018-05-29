export function sigmoidal(synapse) {
    return 1 / (1 + Math.pow(Math.E, -synapse));
}

export function prime(synapse) {
    const output = sigmoidal(synapse);

    return output - output * output;
}
