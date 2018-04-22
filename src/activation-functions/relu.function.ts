export function reLU(synapse) {
    return synapse >= 0 ? synapse : 0;
}
