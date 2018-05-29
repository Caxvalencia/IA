export namespace ReLU {
    export function activation(synapse) {
        return synapse >= 0 ? synapse : 0;
    }
}
