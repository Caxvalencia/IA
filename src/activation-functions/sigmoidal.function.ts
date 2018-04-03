export function sigmoidal(synapse) {
    return 1 / (1 + Math.pow(Math.E, -this.synapse));
}
