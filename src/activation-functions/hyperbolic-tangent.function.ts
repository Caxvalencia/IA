export function hyperbolicTangent(synapse) {
    return Math.tanh(synapse);
}

export function prime(synapse) {
    const output = hyperbolicTangent(synapse);

    return 1.0 - output * output;
}
